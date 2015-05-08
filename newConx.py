#=============================================================================
# newConx.py
#
# James B. Marshall
# version 12/11/2006

#=============================================================================

import math, string, os
from graphics import *
from conx import *

def validateShape(values, shape, default):
    if shape is 0: shape = default
    if shape is None:
        shape = (1, len(values))
    elif type(shape) is int and 0 < shape <= len(values):
        cols = shape
        rows = int(math.ceil(float(len(values))/shape))
        shape = (rows, cols)
    elif type(shape) is tuple and len(shape) == 1:
        shape = (1, shape[0])
    assert type(shape) is tuple and len(shape) == 2, 'invalid shape: %s' % shape
    (rows, cols) = shape
    assert rows * cols == len(values), \
           "can't display %d values with shape %s" % (len(values), shape)
    return shape

def validateScale(scale, default):
    if scale is 0: scale = default
    assert type(scale) is int and scale > 0, 'invalid scale: %s' % scale
    return scale

class PGM(GraphWin):

    def __init__(self, file=None, values=None, title='', shape=0, scale=10, highlight=None):
        if file is None and values is not None:
            assert type(values) is list and len(values) > 0, 'a list of values is required'
            for v in values:
                assert 0 <= v <= 1, 'image values must be in range 0-1'
            shape = validateShape(values, shape, None)
            self.rows, self.cols = shape
            self.normalized = values
            self.raw = [int(100*v) for v in values]
            self.maxval = 100
        elif file is not None and values is None:
            assert shape is 0, 'shape is determined by PGM file'
            f = open(file)
            pgmType = f.readline().strip()
            if pgmType != 'P5':
                raise IOError, 'file is not a raw PGM file'
            title = os.path.basename(file)
            self.cols, self.rows = [int(v) for v in f.readline().split()]
            self.maxval = int(f.readline().strip())
            self.raw = [ord(v) for v in f.read()]
            for v in self.raw:
                assert 0 <= v <= self.maxval, 'incorrect PGM file format'
            self.normalized = [float(v)/self.maxval for v in self.raw]
            f.close()
        else:
            raise AttributeError, 'must specify file=<filename> or values=<vector>'
        if highlight is not None:
            assert type(highlight) is int and self.rows == 1, \
                   'cannot highlight images with more than one row'
            assert 0 <= highlight < self.cols, 'highlight out of range'
        scale = validateScale(scale, None)
        GraphWin.__init__(self, title, self.cols*scale, self.rows*scale)
        self.title = title
        self.rectangles = []
        for i in xrange(len(self.normalized)):
            x = (i % self.cols) * scale
            y = (i / self.cols) * scale
            grayLevel = int(100 * self.normalized[i])
            fColor = oColor = 'gray%d' % grayLevel
            if self.rows == 1: oColor = 'black'
            r = self.create_rectangle(x, y, x+scale, y+scale, outline=oColor, fill=fColor)
            self.rectangles.append(r)
        if highlight is not None:
            x = highlight * scale + 1
            self.create_rectangle(x, 1, x+scale-1, scale, outline='red')

    def __str__(self):
        s = '\ntitle:  %s\n' % self.title
        s += 'size:   %d rows, %d cols\n' % (self.rows, self.cols)
        s += 'maxval: %d\n' % self.maxval
        border = '+%s+\n' % ('-' * (2 * self.cols + 1))
        s += border
        palette = ' .,:+*O8@@'
        for r in xrange(self.rows):
            s += '| '
            for c in xrange(self.cols):
                i = r * self.cols + c
                if i >= len(self.normalized):
                    s += '  '
                else:
                    s += '%s ' % palette[int(self.normalized[i]*(len(palette)-1))]
            s = s + '|\n'
        s += border
        return s

    def setTitle(self, title):
        self.winfo_toplevel().title(title)
        self.title = title

    def updateImage(self, newValues):
        assert len(newValues) == len(self.rectangles), 'wrong number of values'
        for i in xrange(len(newValues)):
            assert 0 <= newValues[i] <= 1, 'image values must be in range 0-1'
            grayLevel = int(100 * newValues[i])
            fColor = oColor = 'gray%d' % grayLevel
            if self.rows == 1: oColor = 'black'
            self.itemconfigure(self.rectangles[i], fill=fColor, outline=oColor)
        self.update_idletasks()
        self.normalized = newValues
        self.raw = [int(100*v) for v in self.normalized]
        self.maxval = 100

    def invert(self):
        self.raw = [self.maxval-v for v in self.raw]
        self.normalized = [float(v)/self.maxval for v in self.raw]
        for i in xrange(len(self.normalized)):
            grayLevel = int(100 * self.normalized[i])
            fColor = oColor = 'gray%d' % grayLevel
            if self.rows == 1: oColor = 'black'
            self.itemconfigure(self.rectangles[i], fill=fColor, outline=oColor)
        self.update_idletasks()

    def saveImage(self, pathname):
        f = open(pathname, mode='w')
        f.write('P5\n')
        f.write('%d %d\n' % (self.cols, self.rows))
        f.write('%d\n' % self.maxval)
        padding = []
        if len(self.raw) < self.rows * self.cols:
            padding = [0] * (self.rows * self.cols - len(self.raw))
        f.write('%s' % string.join([chr(v) for v in self.raw + padding], ''))
        f.close()

#------------------------------------------------------------------------

class WeightDisplay(PGM):

    def __init__(self, connection, i, shape, scale, showBias):
        # need to save connection object itself in order for initialize() to work
        self.connection = connection
        self.i = i
        self.showBias = showBias
        weights = [row[i] for row in self.connection.weight]
        if showBias:
            bias = connection.toLayer.weight[i]
            weights = [bias] + weights
            highlight = 0
        else:
            highlight = None
        maxMagnitude = max(1.0, abs(max(weights)), abs(min(weights)))
        scaledWeights = [(w + maxMagnitude)/(2 * maxMagnitude) for w in weights]
        PGM.__init__(self, values=scaledWeights, shape=shape, scale=scale, highlight=highlight)

    def update(self):
        weights = [row[self.i] for row in self.connection.weight]
        if self.showBias:
            bias = self.connection.toLayer.weight[self.i]
            weights = [bias] + weights
        maxMagnitude = max(1.0, abs(max(weights)), abs(min(weights)))
        scaledWeights = [(w + maxMagnitude)/(2 * maxMagnitude) for w in weights]
        self.updateImage(scaledWeights)

#------------------------------------------------------------------------

class ActivationDisplay(PGM):

    def __init__(self, layer, units, shape, scale):
        # units is a list of unit index numbers
        self.layer = layer
        self.units = units
        self.shape = shape
        activations = [layer.activation[i] for i in units]
        PGM.__init__(self, values=activations, shape=shape, scale=scale)

    def update(self):
        # assumes activations are always in range 0-1
        activations = [self.layer.activation[i] for i in self.units]
        self.updateImage(activations)

#------------------------------------------------------------------------

class BackpropNetwork(Network):

    def __init__(self):
        Network.__init__(self)
        self.weightDisplays = []
        self.activationDisplays = []
        self.defaultShape = None
        self.defaultScale = 10
        self.testInputs = []
        self.testTargets = []
        self.resetLimit = 1

    # sweeps argument is in conx version 233, but not 154
    def train(self, sweeps=None):
        if sweeps is None:
            self.resetEpoch = 10000
        else:
            assert type(sweeps) is int and sweeps > 0, 'invalid number of sweeps'
            self.resetEpoch = sweeps
        Network.train(self)

    def showData(self):
        print "%d training patterns, %d test patterns" % (len(self.inputs), len(self.testInputs))

    def swapData(self, verbose=True):
        # swap training and testing datasets
        if verbose:
            print "Swapping training and testing sets..."
        self.inputs, self.testInputs = self.testInputs, self.inputs
        self.targets, self.testTargets = self.testTargets, self.targets
        if verbose:
            self.showData()

    # splitData takes a percentage from 0 to 100 as input, called trainingPortion,
    # and partitions the data into a training set and a testing set.
    def splitData(self, trainingPortion=None):
        if type(trainingPortion) not in [int, float] or not 0 <= trainingPortion <= 100:
            print 'percentage of dataset to train on is required (0-100)'
            return
        patterns = zip(self.inputs + self.testInputs, self.targets + self.testTargets)
        assert len(patterns) > 0, "no dataset"
        print "Randomly shuffling data patterns..."
        random.shuffle(patterns)
        numTraining = int(math.ceil(trainingPortion / 100.0 * len(patterns)))
        self.inputs = [i for (i, t) in patterns[:numTraining]]
        self.targets = [t for (i, t) in patterns[:numTraining]]
        self.testInputs = [i for (i, t) in patterns[numTraining:]]
        self.testTargets = [t for (i, t) in patterns[numTraining:]]
        print "%d training patterns, %d test patterns" % (len(self.inputs), len(self.testInputs))

    def showPerformance(self):
        if len(self.inputs) ==  0:
            print 'no patterns to test'
            return
        if 'classify' in dir(self):
            self.countWrong = True
            self.numRight = 0
            self.numWrong = 0
        else:
            self.countWrong = False
        learn, order, interact = self.learning, self.orderedInputs, self.interactive
        self.setLearning(0)
        self.setInteractive(1)
        # this forces sweep to go through the dataset patterns in order
        self.setOrderedInputs(1)
        self.sweep()
        if self.countWrong and self.interactive:
            print 'Got %d right, %d wrong' % (self.numRight, self.numWrong)
        # restore previous values
        self.setLearning(learn)
        self.setOrderedInputs(order)
        self.setInteractive(interact)

    def showGeneralization(self):
        self.swapData(verbose=False)
        self.showPerformance()
        self.swapData(verbose=False)

    def updateGraphics(self):
        self.weightDisplays = [wd for wd in self.weightDisplays]
        self.activationDisplays = [ad for ad in self.activationDisplays]
        for wd in self.weightDisplays:
            wd.update()
        for ad in self.activationDisplays:
            ad.update()

    def showWeights(self, layerName, i=None, shape=0, scale=0, showBias=False):
        assert layerName in [layer.name for layer in self.layers], 'no such layer: %s' % layerName
        assert self[layerName].type in ['Hidden', 'Output'], \
               'showWeights only works for hidden or output units'
        toLayer = self[layerName]
        if layerName == 'hidden':
            fromLayer = self['input']
        elif layerName == 'output':
            fromLayer = self['hidden']
        shape = validateShape(fromLayer, shape, self.defaultShape)
        scale = validateScale(scale, self.defaultScale)
        if showBias:
            (rows, cols) = shape
            assert rows == 1, 'shape %s cannot display bias value' % shape
            shape = (rows, cols+1)
        connection = self.getConnection(fromLayer.name, toLayer.name)
        if i is None:
            units = range(toLayer.size)
        else:
            assert 0 <= i < toLayer.size, 'invalid %s unit number: %d' % (toLayer.name, i)
            units = [i]
        for i in units:
            wd = WeightDisplay(connection, i, shape, scale, showBias)
            wd.setTitle('%s[%d] weights' % (toLayer.name, i))
            self.weightDisplays.append(wd)

    def showActivations(self, layerName, units='ALL', shape=0, scale=0):
        assert layerName in [layer.name for layer in self.layers], 'no such layer: %s' % layerName
        layer = self[layerName]
        if units is 'ALL':
            units = range(layer.size)
        else:
            assert type(units) is list and len(units) > 0, 'a list of unit numbers is required'
            for i in units:
                assert type(i) is int and 0 <= i < layer.size, \
                       'invalid %s unit number: %s' % (layer.name, i)
        shape = validateShape(units, shape, self.defaultShape)
        scale = validateScale(scale, self.defaultScale)
        ad = ActivationDisplay(layer, units, shape, scale)
        if units == range(units[0], units[-1]+1):
            # units are contiguous
            unitNums = '%d-%d' % (units[0], units[-1])
        else:
            unitNums = string.join([str(i) for i in units], ', ')
        ad.setTitle('%s units %s' % (layer.name, unitNums))
        self.activationDisplays.append(ad)

    # displays a specific input pattern graphically in a popup window
    def showInput(self, inputNumber, shape=0, scale=0):
        assert len(self.inputs) > 0, 'no input patterns are currently defined'
        assert 0 <= inputNumber < len(self.inputs), \
               "input number must be in range 0-%d" % (len(self.inputs) - 1)
        pattern = self.inputs[inputNumber]
        shape = validateShape(pattern, shape, self.defaultShape)
        scale = validateScale(scale, self.defaultScale)
        title = 'input #%d' % inputNumber
        pgm = PGM(values=pattern, title=title, shape=shape, scale=scale)

    def showPattern(self, pattern, title='', shape=0, scale=0):
        assert type(pattern) is list and len(pattern) > 0, 'invalid pattern'
        shape = validateShape(pattern, shape, self.defaultShape)
        scale = validateScale(scale, self.defaultScale)
        pgm = PGM(values=pattern, title=title, shape=shape, scale=scale)

    def showPatterns(self, patterns, shape=0, scale=0):
        assert type(patterns) is list and len(patterns) > 0, 'a list of patterns is required'
        if type(patterns[0]) is not list:
            # assume patterns is really a single pattern
            self.showPattern(patterns, shape=shape, scale=scale)
        else:
            shape = validateShape(patterns[0], shape, self.defaultShape)
            scale = validateScale(scale, self.defaultScale)
            pgm = PGM(values=patterns[0], shape=shape, scale=scale)
            for pattern in patterns[1:]:
                answer = raw_input('<enter> to continue, <q> to quit... ')
                if answer in ['Q', 'q']:
                    pgm.close()
                    return
                validateShape(pattern, shape, self.defaultShape)
                if pgm.closed: return
                pgm.updateImage(pattern)
            raw_input('<enter> to close... ')
            pgm.close()

    def initialize(self):
        Network.initialize(self)
        self.updateGraphics()

    def apply(self, pattern):
        # save mode
        interactive = self.interactive
        self.setInteractive(1)
        output = self.propagate(input=pattern)
        print 'output is [%s]' % pretty(output)
        # restore mode
        self.setInteractive(interactive)

    def propagate(self, *arg, **kw):
        """ Propagates activation through the network."""
        output = Network.propagate(self, *arg, **kw)
        if self.interactive:
            self.updateGraphics()
        # FIX SUGGESTED BY JIM
        outputPattern = [float(x) for x in output]
        return outputPattern

    def sweep(self):
        result = Network.sweep(self)
        self.updateGraphics()
        return result

    def loadWeightsFromFile(self, filename):
        Network.loadWeightsFromFile(self, filename)
        self.updateGraphics()

    # saveHiddenReps creates a file containing the hidden layer representations
    # generated by the network for all of the input patterns in the dataset,
    # and, if the classify method is present, a parallel file of labels
    # corresponding to the network's classification of each input.
    def saveHiddenReps(self, filename):
        if len(self.inputs) == 0:
            print 'no input patterns available'
            return
        learn, order, interact = self.learning, self.orderedInputs, self.interactive
        self.setLearning(0)
        self.setInteractive(0)
        # this forces sweep to go through the dataset patterns in order
        self.setOrderedInputs(1)
        # record the internal hidden layer patterns in a log file
        logfile = filename + '.hiddens'
        hidden = self.getLayer('hidden')
        hidden.setLog(logfile)
        self.sweep()
        hidden.closeLog()
        print '%d hidden layer patterns saved in %s' % (len(self.inputs), logfile)
        if 'classify' in dir(self):
            # create a parallel file of classification labels
            labelfile = filename + '.labels'
            f = open(labelfile, 'w')
            for pattern in self.inputs:
                output = self.propagate(input=pattern)
                label = self.classify(output)
                f.write('%s\n' % label)
            f.close()
            print '%d classifications saved in %s' % (len(self.inputs), labelfile)
        # restore previous values
        self.setLearning(learn)
        self.setOrderedInputs(order)
        self.setInteractive(interact)
        self.updateGraphics()

    def printWeights(self, fromLayerName, toLayerName, whole=2, frac=4, color=False):
        layerNames = [layer.name for layer in self.layers]
        assert fromLayerName in layerNames, 'no such layer: %s' % fromLayerName
        assert toLayerName in layerNames, 'no such layer: %s' % toLayerName
        connection = self.getConnection(fromLayerName, toLayerName)
        numFrom = connection.fromLayer.size
        numTo = connection.toLayer.size
        fromName = connection.fromLayer.name
        toName = connection.toLayer.name
        fromLabelWidth = len('%s[%d]' % (fromName, numFrom - 1))
        toLabelWidth = whole + frac + 2
        print 'Weights from %s to %s:' % (fromName, toName)
        print ' ' * fromLabelWidth,
        for i in xrange(numTo):
            label = '%s[%d]' % (toName[0:min(3, len(toName))], i)
            print self.colorize('gray', '%*s' % (toLabelWidth, label), color=color),
        print self.colorize('red', '\n%-*s' % (fromLabelWidth, 'bias'), color=color),
        for i in xrange(numTo):
            bias = connection.toLayer.weight[i]
            print self.colorize('red', self.formatReal(bias, whole, frac), color=color),
        print
        for j in xrange(numFrom):
            label = '%s[%d]' % (fromName, j)
            print self.colorize('gray', '%-*s' % (fromLabelWidth, label), color=color),
            for i in xrange(numTo):
                w = connection.weight[j][i]
                print self.formatReal(w, whole, frac),
            print
        print

    def formatReal(self, value, maxWholeSize, maxFracSize):
        maxSize = maxWholeSize + maxFracSize + 2
        wholeSize = len(str(int(abs(value))))
        if wholeSize > maxWholeSize:
            maxFracSize = max(0, maxFracSize - (wholeSize - maxWholeSize))
        if wholeSize > maxSize - 2:
            if value > 0:
                s = '>+' + ('9' * (maxSize - 2))
            else:
                s = '<-' + ('9' * (maxSize - 2))
        else:
            s = '%+*.*f' % (maxSize, maxFracSize, value)
        return s

    def colorize(self, colorName, string, color=True):
        colors = {'red': 31, 'green': 32, 'brown': 33, 'blue': 34, 'magenta': 35,
                  'darkblue': 36, 'gray': 37, 'underline': 38, 'white': 40, 'invRed': 41,
                  'invGreen': 42, 'invBrown': 43, 'invBlue': 44, 'invMagenta': 45}
        if color is False or colorName not in colors:
            return string
        else:
            return '\033[01;%dm%s\033[0m' % (colors[colorName], string)

    # overrides conx version (this version produces simpler output) - 11/29/06 
    def reportEpoch(self, epoch, tssErr, totalCorrect, totalCount, rmsErr, pcorrect = {}):
        # pcorrect is a dict of layer error/correct data:
        #   {layerName: [correct, total, pattern correct, pattern total]...}
        self.Print('Epoch #%6d | TSS Error: %.4f | Correct: %.4f' % \
                   (epoch, tssErr, totalCorrect * 1.0 / totalCount))
        for layerName in pcorrect:
            self[layerName].pcorrect = pcorrect[layerName][2]
            self[layerName].ptotal = pcorrect[layerName][3]
            self[layerName].correct = float(pcorrect[layerName][2]) / pcorrect[layerName][3]
        sys.stdout.flush()

    # overrides conx version (this version produces simpler output) - 11/29/06 
    def reportFinal(self, epoch, tssErr, totalCorrect, totalCount, rmsErr, pcorrect = {}):
        # pcorrect is a dict of layer error/correct data:
        #   {layerName: [correct, total, pattern correct, pattern total]...}
        self.Print('Final #%6d | TSS Error: %.4f | Correct: %.4f' % \
                   (epoch-1, tssErr, totalCorrect * 1.0 / totalCount))
        for layerName in pcorrect:
            self[layerName].pcorrect = pcorrect[layerName][2]
            self[layerName].ptotal = pcorrect[layerName][3]
            self[layerName].correct = float(pcorrect[layerName][2]) / pcorrect[layerName][3]
        sys.stdout.flush()

    # overrides Network method - removes conx quit option
    def prompt(self):
        print "==================================="
        input = raw_input('<enter> to continue, <q> to quit... ')
        #input = raw_input('--More-- [<enter> or <g>o...] ')
        #if input in ['G', 'g']:
        if input in ['Q', 'q']:
            self.interactive = 0

    # overrides Network method
    def display(self):
        """Displays the network to the screen."""
        size = range(len(self.layers))
        size.reverse()
        for i in size:
            layer = self.layers[i]
            if layer.active:
                print '%s layer (size %d)' % (layer.name, layer.size)
                tlabel, olabel = '', ''
                if (layer.type == 'Output'):
                    if self.countWrong:
                        tlabel = ' (%s)' % self.classify(layer.target.tolist())
                        olabel = ' (%s)' % self.classify(layer.activation.tolist())
                        if olabel == tlabel:
                            self.numRight += 1
                        if olabel != tlabel:
                            olabel += '  *** WRONG ***'
                            self.numWrong += 1
                    print 'Target    : %s%s' % (pretty(layer.target, max=15), tlabel)
                print 'Activation: %s%s' % (pretty(layer.activation, max=15), olabel)
                print
                #print "-----------------------------------"

    def addLayers(self, *sizes):
        assert 2 <= len(sizes) <= 3, 'only works for 2 or 3 layers'
        if len(sizes) == 2:
            (numInputs, numOutputs) = sizes
            self.addLayer('input', numInputs)
            self.addLayer('output', numOutputs)
            self.connect('input', 'output')
        else:
            (numInputs, numHiddens, numOutputs) = sizes
            self.addLayer('input', numInputs)
            self.addLayer('hidden', numHiddens)
            self.addLayer('output', numOutputs)
            self.connect('input', 'hidden')
            self.connect('hidden', 'output')
   

def pretty(values, max=0):
    if max > 0 and len(values) > max:
        return string.join(['%.2f' % v for v in values[0:max]]) + ' ...'
    else:
        return string.join(['%.2f' % v for v in values])
