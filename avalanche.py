#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:01:24 2018

@author: patrickschulz
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gdal
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import elevation 
import os
from scipy.integrate import ode 

print('')
print('Welcome to my Avalanche App!')
print('')
print('This application takes a square region along with weather data and finds the probability of an avalanche occurring in that region.')
print('')
print('Please enter the following information:')
#minX = float(input('min longitude:'))
#maxX = float(input('max longitude:'))
#minY = float(input('min latitude:'))
#maxY = float(input('max latitude:'))
#name = input('Name this area:')
name = 'Park city'
def getTif(minX,maxX,minY,maxY,fname):
    
    #print(int(round(maxX - minX, 2)), int(round(maxY - minY, 2)))
    
    if minX > maxX or minY > maxY:
        print('')
        print('**Error: Check Your latitude and longitude bounds!**')
        print('')
    elif round(maxX - minX,3) != round(maxY - minY,3):
        print('')
        print('**Error: Your bounds must be a square region!**')
        print('')
    else:
        dem_path = fname
        output = os.getcwd() + dem_path
        elevation.clip(bounds=(minX,minY,maxX,maxY), output=output)
        elevation.clean()
        return(output,minX,maxX,minY,maxY)
    

#Snowbird -111.7, -111.6, 40.5, 40.6
#Park City -111.58 -111.52 40.6 40.66
(filepath,minX,maxX,minY,maxY) = getTif(-111.58, -111.52, 40.6, 40.66,'/testTwin-DEM.tif')
#(filepath,minX,maxX,minY,maxY) = getTif(minX,maxX,minY,maxY,'/testTwin-DEM.tif')

#filepath = '/Users/patrickschulz/Documents/Programing/Python/Avalanche_app/testTwin-DEM.tif'
#-----------------------------------------------------------------------------

def getElevation(filename):
    gdal.UseExceptions()

    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    elev = band.ReadAsArray()
    
    grad=np.gradient(elev)
    slope = grad[1]/grad[0]
    angle = np.arctan(slope)
    deg = np.abs(angle*(180/np.pi))
    
    return(elev, deg)

(elev,deg) = getElevation(filepath)
Z = np.flip(elev,0)
#-----------------------------------------------------------------------------

def getDataNWS(filePath):  
#reads in the data files to python and orders them into float lists 

    
    date = []
    low = []
    avgT = []
    high = []
    percip = []
    snow = []
    humid = []
    press = []
    wind = []
    
    with open(filePath, "r") as F:
            for line in F:
                array = line.split()
                date.append(float(array[0]))
                low.append(float(array[1]))
                avgT.append(float(array[2]))
                high.append(float(array[3]))
                percip.append(float(array[4]))
                snow.append(float(array[5]))
                humid.append(float(array[6]))
                press.append(float(array[7]))
                wind.append(float(array[8]))
                
                
    return(date,low,avgT,high,percip,snow,humid,press,wind)

#-----------------------------------------------------------------------------

def getSeason():
#adds all the data (temp in c) from all the months starting from the begining of november
#or whenever the first snowday was 

    
    Nov17 = getDataNWS("/Users/patrickschulz/Documents/Programing/Python/KSLC_Data/KSLC_November_2017.txt")
    Dec17 = getDataNWS("/Users/patrickschulz/Documents/Programing/Python/KSLC_Data/KSLC_December_2017.txt")
    Jan18 = getDataNWS("/Users/patrickschulz/Documents/Programing/Python/KSLC_Data/KSLC_January_2018.txt")
    Feb18 = getDataNWS("/Users/patrickschulz/Documents/Programing/Python/KSLC_Data/KSLC_February_2018.txt")
    Mar18 = getDataNWS("/Users/patrickschulz/Documents/Programing/Python/KSLC_Data/KSLC_March_2018.txt")
    
    
    avgT = Nov17[2]+Dec17[2]+Jan18[2]+Feb18[2]+Mar18[2]
    percip = Nov17[4]+Dec17[4]+Jan18[4]+Feb18[4]+Mar18[4]
    snow = Nov17[5]+Dec17[5]+Jan18[5]+Feb18[5]+Mar18[5]
    humid = Nov17[6]+Dec17[6]+Jan18[6]+Feb18[6]+Mar18[6]
    press = Nov17[7]+Dec17[7]+Jan18[7]+Feb18[7]+Mar18[7]
    wind = Nov17[8]+Dec17[8]+Jan18[8]+Feb18[8]+Mar18[8]


    for i in range(150):
        if snow[i] > 0: 
            break 
        
        
    del avgT[0:i]
    del percip[0:i] 
    del snow[0:i] 
    del humid[0:i]
    del press[0:i]
    del wind[0:i]
    
    temp = []
    for i in range(len(avgT)):
        temp.append((avgT[i]-32)*(5/9)) #changes temp to C˚   
        wind[i] = (wind[i]*0.44704) #changes wind from mph to m/s
        snow[i] = (snow[i]*25.4) 
        percip[i] = (percip[i]*25.4) #changes in to mm
        humid[i] = (humid[i]/100) #makes humidity a percenatage
    
    
    
    return(temp, percip, snow, humid, press, wind, i)
    
#-----------------------------------------------------------------------------

def getDelta(): 

    dR = 1.8 #absolute value of hardness difference 
    dE = 0.6 #absolute value of grainsize difference in mm
    return(dR,dE)
    
def getRho():
    tau = 2
    rho = 3 
    return(tau,rho
           )
def snowpackModel():
    (temp, percip, snow, humid, press, wind, i) = getSeason()
    tempK = np.ndarray([len(temp)])
    snowpackSet = np.ndarray([5,len(snow)])
    snowpack = np.ndarray([2,len(snow)])
    for i in range(len(temp)):
        if snow[i] > 0:
            tempK[i].append(temp[i] - 273.15)
            snowpack[0,i] = i
            snowpack[1,i] = snow[i]
            snowpackSet[0,i] = temp[i]  
            snowpackSet[1,i] = percip[i]  
            snowpackSet[2,i] = humid[i]   
            snowpackSet[3,i] = press[i]   
            snowpackSet[4,i] = wind[i]   
    tau = 2
    rho = 3 
    eta = np.ndarray([len(snowpack)])
    dayCounter = 0
    alb = np.ndarray([len(snowpack)])
    snowHeight = np.array([len(snowpack)])
    snowHeight[0] = 0
    for i in range (len(snowpackSet)):
        if snowpackSet[0,i] > 0:
            dayCounter = dayCounter+1
            eta[i] = float(5.38*10**-3*np.exp(0.024*0.916+6042))
            alb[i] = 0.88-6*10**-3
        if snowpackSet[0,i] < 0:
            alb[i] = float(0.8-0.03*temp[i]-1.74*10**-3*(temp[i]**2)-1.14*10**-4(temp[i]**3))
            eta[i] = float(5.38*10**-3*np.exp(0.024*1+6042/temp[i]))
    dR = 1.8  
    dE = 0.6       
    def ode1(alb,t):
        DalbedoDt = np.array([len(alb)])
        for i in range(len(eta)):
            DalbedoDt[i] = eta[i]*dayCounter
        return(DalbedoDt)
        
    for i in range(len(snowHeight)):
        snowHeight[i] = (ode(ode1(alb,dayCounter)))
    
    def ode2(snowHeight,eta):
        DhDeta = np.array([len(snowHeight)])
        for i in range(len(eta)):
            DhDeta[i] = eta[i]*snowHeight[i]
        return(DhDeta)
    
    rho = ode(ode1(snowHeight,eta),dayCounter)
    tau = ode(ode2(snowHeight,eta),dayCounter)
    return(rho,tau,dR,dE)


    
    
#-----------------------------------------------------------------------------


def getSK(deg):
    
    (tau,rho)=getRho()
    DeltaTau = float(98.1*(np.cos(46)/0.8)*np.sin(46)**2*(np.sin(84)))     
    
    tauXZ = np.ndarray([len(deg),len(deg)])
    SK = np.ndarray([len(deg),len(deg)])
    for i in range (len(deg)):
        for j in range (len(deg[i])):
            psi = deg[i,j]
            if psi > 70:
                SK[i,j] = 0.50
            elif psi < 25:
                SK[i,j] = 0.1
            else:
                h = np.tan(psi) 
                tauXZ[i,j] = float(rho*9.81*h*np.sin(psi)*np.cos(psi))
                SK[i,j] = float(abs(tau/(tauXZ[i,j]+DeltaTau)))
    
    return(SK)

#-----------------------------------------------------------------------------

def getSSI(deg):
    
    (deltR,deltE) = getDelta()
    
    if deltR >= 1.5 and deltE >= 0.5:
        D = 0
    elif deltR < 1.5 or deltE < 0.5:
        D = 1

    SK = getSK(deg)
    SSI = np.ndarray([len(deg),len(deg)])
    for i in range (len(deg)):
        for j in range (len(deg[i])):
            SSI[i,j] = SK[i,j]+D
    return(SSI)

    
ssi = getSSI(deg)
sk = getSK(deg)
#-----------------------------------------------------------------------------

def avalanche(deg):
    
    rgbs = [(0, .6, 0), (1, 1, 0), (1, 0, 0)]
    # 0 =green, 1=yellow, 2=red
    cmap_name = 'my_list'
    ava = LinearSegmentedColormap.from_list(cmap_name, rgbs, N=3)
    
    ssi = getSSI(deg)
    sk = getSK(deg)
    
    overlay = np.ndarray([len(deg),len(deg)])
    for i in range (len(deg)):
        for j in range (len(deg)):
            if sk[i][j] >= 0.45: #good conditions
                overlay[i][j] = Z.min()+2 #green
            elif sk[i][j] < 0.45 and ssi[i][j] >= 1.32:
                overlay[i][j] = Z.min()+1 #yellow
            else:
                overlay[i][j] = Z.min()+0 #red
    
    return(overlay,ava)


(overlay,ava) = avalanche(deg)

#-----------------------------------------------------------------------------

a = plt.subplot(121)
a.imshow(deg, cmap='gray', alpha=0.95,interpolation='bicubic', extent=[minX,maxX,minY,maxY])
a.imshow(elev, cmap='binary', alpha=0.9,interpolation='bicubic', extent=[minX,maxX,minY,maxY])
plt.xticks(rotation=40)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Your Mountain')
plt.grid(True)


b = plt.subplot(122,sharex=a, sharey=a)
risk = plt.imshow(overlay, cmap=ava,interpolation='bicubic',extent=[minX,maxX,minY,maxY])
greenP = mpatches.Patch(color='green', label='Low Risk - 5%')
yellowP = mpatches.Patch(color='yellow', label='Medium Risk - 35%')
redP = mpatches.Patch(color='red', label='High Risk - 70%')
lines = [greenP, yellowP, redP]
labels = [line.get_label() for line in lines]
plt.setp(b.get_yticklabels(), visible=False)
plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
plt.xticks(rotation=40)
plt.title('Avalanche Risk')
plt.xlabel('Longitude')
plt.grid(True)

plt.suptitle('2D Map of ' + name)
plt.subplots_adjust(right=0.75, bottom=0.3,top=0.7, wspace=0.1)




#-----------------------------------------------------------------------------

X = np.linspace(minX,maxX,len(elev))
Y = np.linspace(minY,maxY,len(elev))
X, Y = np.meshgrid(X, Y)
Z2 = np.flip(overlay,0)

fig = plt.figure()
ax = fig.gca(projection= '3d')
plt.figure(2)
ax.plot_surface(X, Y, Z, cmap='binary', alpha=0.8)
ax.contourf(X, Y, Z2, zdir='z',offset=Z.min(), cmap=ava)
greenP = mpatches.Patch(color='green', label='Low Risk - 5%')
yellowP = mpatches.Patch(color='yellow', label='Medium Risk - 35%')
redP = mpatches.Patch(color='red', label='High Risk - 70%')
lines = [greenP, yellowP, redP]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation [m]')
ax.set_title('3D Map of Avalanche Risk of '+name)
plt.subplots_adjust(left=0,right=0.75)
plt.show()









