# Eq 3.2-2: s = c*log(1+r)
# Eq 3.2-3: s = c*r**gamma
# Fig0308.tif

from PIL import Image
import math, os
import matplotlib.pyplot as plt


class A1Part1:
    def __init__(self):
        self.name = None
        self.mode = None
        self.size = None
        self.format = None
        self.pixel = None

    # get all the attributes of the image information
    def getInfo(self, inputFile):
        self.name, ext = os.path.splitext(inputFile)
        im = Image.open(inputFile)
        self.mode, self.size, self.format = im.mode, im.size, im.format
        self.pixel = list(im.getdata())

    # define log function
    def logFunc(self, px, c):
        return (c * math.log((1+px), 2))

    # define power law function
    def powerLaw(self, px, c, gamma):
        return (c * px**gamma)

    # print transformation function
    def drawFunc(self, func, c=1.0, gamma=1.0):
        oldIntensities = [p/255.0 for p in range(256)]
        if func == 'logFunc':
            newIntensities = [self.logFunc(p, c) for p in oldIntensities]
        else:
            newIntensities = [self.powerLaw(p, c, gamma) for p in oldIntensities]
        plt.plot(oldIntensities, newIntensities)
        plt.xlabel('Old Intensities')
        plt.ylabel('Transformed Intensities')
        if func == 'logFunc':
            plt.title(func+' Transformation'+' c='+str(c))
        else:
            plt.title(func+' Transformation'+' c='+str(c)+' gamma='+str(gamma))
        plt.show()
        return

    # save the original pixel information in txt file
    def printOriginalPixel(self):
        l, w = self.size
        data = [self.pixel[x:x+l] for x in xrange(0, len(self.pixel), l)]
        f = open(self.name+'_Original_.txt', 'w')
        for subLst in data:
            f.write(str(subLst)+'\n')
        f.close()
        return

    # enhance the image by applying function on pixels
    def pixelEnhancement(self, func, thumbnail='', c=70, gamma=1):
        if func == 'logFunc':
            thumbnail += '_log_'
            newPixel = [self.logFunc(px/255.0, c)*255.0 for px in self.pixel]
        else:
            thumbnail += '_power_'
            newPixel = [self.powerLaw(px/255.0, c, gamma)*255.0 for px in self.pixel]
        outputFile = self.name + thumbnail + '.' + self.format
        newIm = Image.new(self.mode, self.size)
        newIm.putdata(newPixel)
        newIm.save(outputFile)

    # run several rounds of testing cases
    def runTests(self, funcName=''):
        NUM_OF_TESTS = 8
        # run logFunc case
        if funcName == 'logFunc':
            c = 1.5
            for _ in range(NUM_OF_TESTS):
                c += 0.2
                thumbnail = '_c' + str(c)
                self.pixelEnhancement(func=funcName, thumbnail=thumbnail, c=c)
        # run powerLaw case
        else:
            c = 1.0
            gamma = 0.3
            for _ in range(NUM_OF_TESTS):
                for _ in range(NUM_OF_TESTS):
                    c += 0.00
                    thumbnail = '_c' + str(c) + '_gamma' + str(gamma)
                    self.pixelEnhancement(func=funcName, thumbnail=thumbnail, c=c, gamma=gamma)
                c = 1.0
                gamma += 0.05


if __name__ == '__main__':
    sol = A1Part1()
    sol.getInfo('Fig0308.tif')
    sol.printOriginalPixel()
    #sol.drawFunc('logFunc')
    #sol.drawFunc('powerLaw', gamma=0.5)
    sol.runTests('powerLaw')
