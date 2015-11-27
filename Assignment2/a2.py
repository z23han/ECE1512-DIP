'''
Part 1:
    1. Write a program to convert an arbitrary rgb image to a web-safe rgb image
    2. Convert fig-6-8 in tif format to a web-safe rbg image, and convert it back
Part 2:
    1. Convert fig-6-35 to RGB. Histogram-equalize the RGB and convert back to TIF format
    2. Form an average histogram from 3 histogram in 1, and use it as basis to obtain a single histogram
    equalization intensity transformation function. Convert RGB to jpg. Compare.
'''

from PIL import Image
import os, collections
import numpy as np
import matplotlib.pyplot as plt


####################  A2  ##################################
class A2:
    def __init__(self, inputFile):
        self.name, ext = os.path.splitext(inputFile)
        self.im = Image.open(inputFile)
        self.mode, self.size, self.format = self.im.mode, self.im.size, self.im.format
        self.rgb = []
        self.wRGB = []

    def getRGB(self):
        rgb_im = self.im.convert('RGB')
        width, height = rgb_im.size
        pixels = rgb_im.load()
        for i in range(width):
            for j in range(height):
                self.rgb.append(pixels[i, j])

    def getWebSafe(self):
        # 6 possible components (0,51,102,153,204,255)
        for pixels in self.rgb:
            r, g, b = pixels
            wr = self.selectionFunc(r)
            wg= self.selectionFunc(g)
            wb = self.selectionFunc(b)
            self.wRGB.append((wr, wg, wb))

    # helper function for allocating colors into web-safe color range
    def selectionFunc(self, r):
        if r >= 0 and r <= 41:
            return 0
        elif r >= 42 and r <= 84:
            return 51
        elif r >= 85 and r <= 127:
            return 102
        elif r >= 128 and r <= 171:
            return 153
        elif r >= 172 and r <= 212:
            return 204
        else:
            return 255

    def webSafeImage(self):
        newIm = Image.new(self.mode, self.size)
        width, height = self.im.size
        cnt = 0
        for i in range(width):
            for j in range(height):
                newIm.putpixel((i, j), self.wRGB[cnt])
                cnt += 1
        imageName = self.name + '-web-safe.' + self.format
        newIm.save(imageName)

    # histogram-equalization algorithms
    def histAlgs(self, lst):
        # store pixels from lst into a new list
        pixList = list(lst)
        # use dictionary mapping the pixel values to the index they appear
        histDict = collections.defaultdict(list)
        for i, item in enumerate(pixList):
            histDict[item].append(i)
        # get the probability and bins from pixList
        imHist, bins = np.histogram(pixList, bins=list(np.arange(-0.5, 256.5, 1)), normed=True)
        # calculate the cumulative sum from the beginning to the end
        cdf = imHist.cumsum()
        # form the new pixel value list
        newGrayLevel = [round(c*255.0) for c in cdf]
        # form the pixel value probability list
        newImHist = [0.0]*255
        for i in range(len(imHist)):
            # get the corresponding new transformed pixel value
            idx = newGrayLevel[i]
            # add the probability for the pixel value in the new position
            newImHist[int(idx)-1] += imHist[i]
            # for each value stored in the dictionary, update them into new pixel value
            if i in histDict:
                for pixIdx in histDict[i]:
                    pixList[pixIdx] = int(idx) - 1
        return pixList, newGrayLevel

    # simply borrow the above method, but use pre-discovered histogram
    def histAveAlgs(self, lst, histogram):
        pixList = list(lst)
        cdf = np.array(histogram).cumsum()
        newGrayLevel = [int(round(c*255)) for c in cdf]
        for i in range(len(pixList)):
            pixList[i] = newGrayLevel[pixList[i]]
        return pixList

    def hisAveFunc(self):
        rList = [p[0] for p in self.rgb]
        gList = [p[1] for p in self.rgb]
        bList = [p[2] for p in self.rgb]
        rImHist, rBins = np.histogram(rList, bins=list(np.arange(-0.5, 256.5, 1)), normed=True)
        gImHist, gBins = np.histogram(gList, bins=list(np.arange(-0.5, 256.5, 1)), normed=True)
        bImHist, bBins = np.histogram(bList, bins=list(np.arange(-0.5, 256.5, 1)), normed=True)
        aveImHist = []
        for i in range(len(rImHist)):
            aveImHist.append((rImHist[i]+gImHist[i]+bImHist[i])/3.0)
        rOutList = self.histAveAlgs(rList, aveImHist)
        gOutList = self.histAveAlgs(gList, aveImHist)
        bOutList = self.histAveAlgs(bList, aveImHist)
        # plot the averaged histogram
        plt.subplot(2,2,1)
        plt.plot(rImHist, color='r')
        plt.title('Red Histogram')
        plt.subplot(2,2,2)
        plt.plot(gImHist, color='g')
        plt.title('Green Histogram')
        plt.subplot(2,2,3)
        plt.plot(bImHist, color='b')
        plt.title('Blue Histogram')
        plt.subplot(2,2,4)
        plt.plot(aveImHist, color='k')
        plt.title('Averaged Histogram')
        plt.show()
        newImage = Image.new(self.mode, self.size)
        width, height = self.im.size
        cnt = 0
        for i in range(width):
            for j in range(height):
                newImage.putpixel((i, j), (rOutList[cnt], gOutList[cnt], bOutList[cnt]))
                cnt += 1
        imageName = self.name + '-hist-ave.' + 'jpg'
        newImage.save(imageName)


    # histogram plotting, input imHist (probability list) and title
    def plotHistogram(self, imHist, title):
        plt.figure()
        plt.bar(list(range(255)), imHist)
        plt.axis([0, 256, 0, 1.0])
        plt.title(title)
        plt.xlabel('Gray Level')
        plt.ylabel('Frequencies')
        plt.show()

    # generate histogram-equalized image
    def histEqual(self):
        rList = [p[0] for p in self.rgb]
        gList = [p[1] for p in self.rgb]
        bList = [p[2] for p in self.rgb]
        histR, _ = self.histAlgs(rList)
        histG, _ = self.histAlgs(gList)
        histB, _ = self.histAlgs(bList)
        # save to the R, G, B histogram-equalization images
        pixelIm = self.im.convert('L')
        rImage = Image.new(pixelIm.mode, pixelIm.size)
        rImage.putdata(histR)
        gImage = Image.new(pixelIm.mode, pixelIm.size)
        gImage.putdata(histG)
        bImage = Image.new(pixelIm.mode, pixelIm.size)
        bImage.putdata(histB)
        newImage = Image.new(self.mode, self.size)
        width, height = self.im.size
        cnt = 0
        for i in range(width):
            for j in range(height):
                newImage.putpixel((i, j), (histR[cnt], histG[cnt], histB[cnt]))
                cnt += 1
        imageName = self.name + '-hist-equal.' + self.format
        newImage.save(imageName)
        rName = self.name + '-R.' + self.format
        gName = self.name + '-G.' + self.format
        bName = self.name + '-B.' + self.format
        rImage.save(rName)
        gImage.save(gName)
        bImage.save(bName)


    # average histogram and use it as basis to get the intensity transformation function
    def transFunc(self):
        rList = [p[0] for p in self.rgb]
        gList = [p[1] for p in self.rgb]
        bList = [p[2] for p in self.rgb]
        # grab the equalization-mapping
        _, rNew = self.histAlgs(rList)
        _, gNew = self.histAlgs(gList)
        _, bNew = self.histAlgs(bList)
        rDict = {}
        gDict = {}
        bDict = {}
        for i in range(len(rNew)):
            rDict[i] = rNew[i]
        for i in range(len(gNew)):
            gDict[i] = gNew[i]
        for i in range(len(bNew)):
            bDict[i] = bNew[i]
        aveMapping = {}
        for i in range(len(rDict)):
            aveMapping[i] = int(round((rDict[i]+gDict[i]+bDict[i])/3.0))
        averList = []
        for i in aveMapping:
            averList.append(aveMapping[i])
        # apply the mapping to all the r, g, b
        transR = [aveMapping[r] for r in rList]
        transG = [aveMapping[g] for g in gList]
        transB = [aveMapping[b] for b in bList]
        # plot transformation function
        plt.plot(rNew, '-r', label='r')
        plt.plot(gNew, '-g', label='g')
        plt.plot(bNew, '-b', label='b')
        plt.plot(averList, '-k', label='average')
        plt.legend(loc='lower right')
        plt.axis([0, 256, 0, 300])
        plt.title('Transformation Function')
        plt.show()
        # create new image
        newImage = Image.new(self.mode, self.size)
        width, height = self.im.size
        cnt = 0
        for i in range(width):
            for j in range(height):
                newImage.putpixel((i, j), (transR[cnt], transG[cnt], transB[cnt]))
                cnt += 1
        imageName = self.name + '-ave-transformed.' + 'jpg'
        newImage.save(imageName, "JPEG")

    # use histogram specification method
    def histSpecification(self):
        rFile = open('outR.txt', 'r')
        gFile = open('outG.txt', 'r')
        bFile = open('outB.txt', 'r')
        rList = []
        gList = []
        bList = []
        for i in range(697):
            # R list
            lineR = rFile.readline()
            lineR = lineR.strip('\n')
            lineListR = lineR.split(',')
            lineListR = [int(r) for r in lineListR]
            rList += lineListR
            # G list
            lineG = gFile.readline()
            lineG = lineG.strip('\n')
            lineListG = lineG.split(',')
            lineListG = [int(g) for g in lineListG]
            gList += lineListG
            # B list
            lineB = bFile.readline()
            lineB = lineB.strip('\n')
            lineListB = lineB.split(',')
            lineListB = [int(b) for b in lineListB]
            bList += lineListB
        # create new image
        newImage = Image.new(self.mode, self.size)
        width, height = self.im.size
        cnt = 0
        for i in range(height):
            for j in range(width):
                newImage.putpixel((j, i), (rList[cnt], gList[cnt], bList[cnt]))
                cnt += 1
        imageName = self.name + '-ave-transformed1.' + 'jpg'
        newImage.save(imageName, "JPEG")
        rFile.close()
        gFile.close()
        bFile.close()



def a2PartA_1():
    a2parta_1 = A2('icon.jpg')
    a2parta_1.getRGB()
    a2parta_1.getWebSafe()
    a2parta_1.webSafeImage()


def a2PartA_2():
    a2parta_2 = A2('Fig0608.tif')
    a2parta_2.getRGB()
    a2parta_2.getWebSafe()
    a2parta_2.webSafeImage()


def a2PartB_1():
    a2Partb_1 = A2('Fig0635.tif')
    a2Partb_1.getRGB()
    a2Partb_1.histEqual()


def a2PartB_2():
    a2Partb_2 = A2('Fig0635.tif')
    a2Partb_2.getRGB()
    a2Partb_2.hisAveFunc()
    #a2Partb_2.transFunc()
    #a2Partb_2.histSpecification()


if __name__ == '__main__':
    a2PartB_2()