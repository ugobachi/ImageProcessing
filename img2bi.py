import cv2
import numpy as np

def bmp_reading(infile, outfile):
    """[summary]
    入力した画像をグレースケール化して二値化、最後に出力
    Args:
        infile (str): 入力画像
        outfile (str): 出力画像名
    """ 
    inputfile = cv2.imread('{}.bmp'.format(infile))
    imgray = cv2.cvtColor(inputfile, cv2.COLOR_BGR2GRAY)
    result = binarization(imgray)
    cv2.imwrite('{}.bmp'.format(outfile), result)
    return 

def binarization(imgray):
    """[summary]
    判別分析法で二値化
    Args:
        imgray ([type]): グレースケール化した画像

    Returns:
        [type]: 最終的に二値化した画像
    """    
    hist = [np.sum(imgray == i) for i in range(256)]
    overall, allpixel = np.mean(imgray), imgray.sum()
    class_var, th = -1, -1

    for threshold in range(256):
        C1, w1 = sum(hist[:threshold]), sum(hist[:threshold]) / allpixel
        C2, w2 = sum(hist[threshold:]), sum(hist[threshold:]) / allpixel

        if C1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,threshold)]) / C1

        if C2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(threshold, 256)]) / C2

        calc = w1*(mu1 - overall)**2 + w2*(mu2 - overall)**2

        if class_var < calc:
            class_var = calc
            th = threshold
    
    imgray[imgray < th] = 0
    imgray[imgray >= th] = 255

    return imgray

if __name__ == "__main__":
    infile = input('input filename : ')
    outfile = input('output filename : ')
    bmp_reading(infile, outfile)