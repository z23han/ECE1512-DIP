import Image, os, collections
import matplotlib.pyplot as plt
import numpy as np


class A1Part2:
    def __init__(self):
        self.im = None
        self.name = None
        self.mode = None
        self.size = None
        self.format = None
        self.pixel = None

    # get all the attributes of the image information
    def getInfo(self, inputFile):
        self.name, ext = os.path.splitext(inputFile)
        self.im = Image.open(inputFile)
        self.mode, self.size, self.format = self.im.mode, self.im.size, self.im.format
        self.pixel = list(self.im.getdata())

    # generate histogram plot
    def generateHist(self):
        im = np.array(self.im.convert('L'))
        # imHist is values of the histogram
        # bins is the bin edges
        imHist, bins = np.histogram(im.flatten(), 256, normed=True)
        # cdf is the cumulative distribution function (cdf)
        cdf = imHist.cumsum()
        # transform the value into [0-255]
        cdf = 255 * cdf / cdf[-1]
        # use linear interpolation of cdf to find new pixel values
        newPixel = np.interp(im.flatten(), bins[:-1], cdf)
        #newPixel = newPixel.reshape(im.shape)
        # make the numpy array to be a list
        newPixel = list(newPixel)
        outputFile = self.name + '_histEqual' + '.' + self.format
        newIm = Image.new(self.mode, self.size)
        newIm.putdata(newPixel)
        newIm.save(outputFile)
        plt.figure(1)
        plt.hist(self.pixel, bins=256, histtype='bar')
        plt.title('Original '+self.name+' Histogram')
        plt.xlabel('Gray Levels')
        plt.ylabel('Frequencies')
        plt.figure(2)
        plt.hist(newPixel, bins=256, histtype='bar')
        plt.title('Modified '+self.name+' Histogram')
        plt.xlabel('Gray Levels')
        plt.ylabel('Frequencies')
        plt.show()
        return newPixel, cdf

    # histogram equalization
    def histEqual(self):
        im = np.array(self.im)
        #### get the pixel values and store into a list
        oldPixel = list(self.im.getdata())
        oldPixel = [round(p) for p in oldPixel]
        ## collections defaultdict stores list dictionary
        histDict = collections.defaultdict(list)
        for i, item in enumerate(oldPixel):
            histDict[item].append(i)
        # input: bins (num of bins)
        # output: imHist (probabilities of each histogram bar), bins (bin edges)
        imHist, bins = np.histogram(im.flatten(), bins=255, normed=True)
        # cdf is the cumulative distribution function (cdf)
        #### histogram equalization method !!!
        cdf = imHist.cumsum()
        newGrayLevel = [round(c*255.0) for c in cdf]
        newImHist = [0.0]*255
        for i in xrange(len(imHist)):
            index = newGrayLevel[i]
            newImHist[int(index)-1] += imHist[i]
            # if index in the histDict, we need to update the oldPixel values
            if i in histDict:
                for pixelIndex in histDict[i]:
                    oldPixel[pixelIndex] = int(index)-1
        #### save the file
        newIm = Image.new(self.mode, self.size)
        newIm.putdata(oldPixel)
        outputFileName = self.name+'_histEqual.'+'png'
        newIm.save(outputFileName)
        #### plot histogram
        plt.figure(1)
        plt.bar(list(range(255)), imHist)
        plt.axis([0, 256, 0, 1.0])
        plt.title('Original '+self.name+' Histogram')
        plt.xlabel('Gray Levels')
        plt.ylabel('Frequencies')
        plt.figure(2)
        plt.bar(list(range(255)), newImHist)
        plt.axis([0, 256, 0, 1.0])
        plt.title('Modified '+self.name+' Histogram')
        plt.xlabel('Gray Levels')
        plt.ylabel('Frequencies')
        #### plot the equalization transformation function
        plt.figure(3)
        plt.plot(newGrayLevel)
        plt.axis([0, 255, 0, 255])
        plt.xlabel('Input Intensity')
        plt.ylabel('Output Intensity')
        plt.title('Histogram Transformation Function')
        plt.show()


if __name__ == '__main__':
    sol = A1Part2()
    sol.getInfo('Fig0308.tif')
    sol.histEqual()