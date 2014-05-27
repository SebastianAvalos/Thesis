# ############################################################################# #
# Modified and implemented Filtersim (R) (Zhang, et.al. 2006) algorithm by      #
# Sebastian Avalos @ University of Chile - 2014 - sebastian.avaloss@gmail.com  #                                                                     #
# ############################################################################# #

import random as ran 
import numpy as np
from filters_2d import *
from scoremap_2d import *
import csv

print "Welcome to the modified FILTERSIM algorithm."
print "This version include local mean drift as a influence parameter."
comenzando = raw_input("Now, press Enter if you want to continue...")
#print

# #######################
# The following parameters are directly expressed by the user instead of bring it as an answer of a algorithm's question
repeticion = 10
ngsx=9
ngsy=9
innerx=9
innery=9
nmgrid=2
minimumdc = 2
w0=0.5
kmeanveces = 5

print
print "Ngsx:", ngsx, " ","Ngsy:", ngsy, " ","Innerx:", innerx, " ","Innery:", innery, " ", "Nmgrid:", nmgrid, "MinimumCD:", minimumdc, "Wo:", w0
# ##################
ncluster = 200  #This number fix the maximum classes
# ##################

if nmgrid == 1:
   w1 = 0.7
   w2 = 0.3
if nmgrid>1:
   w1 = 0.6
   w2 = 0.25
   w3 = 0.15

# #####################################

#repeticion=int(raw_input("Number of simulation:  "))
while repeticion<=0:
   repeticion=int(raw_input("Please put a positive number: "))
print


# Block model inputs in order to generate the training image
print "Opening and reading the block model (as T.I.) parameters file:"
# The file has to be as: parmb.txt
ParBM = np.loadtxt("parmb_2d.txt")
numx = int(ParBM[0])
numy = int(ParBM[1])
tamx = int(ParBM[2])
tamy = int(ParBM[3])
minx = int(ParBM[4])
miny = int(ParBM[5])
print "       . . . Parameters loaded successfully"
print


print "Opening and reading the block model (as T.I.) data file:"
# the file has to be as: mb.txt
BM = np.loadtxt("mb_2d.txt")
TI=np.zeros((numx,numy))
GSvalTI = np.zeros((nmgrid,numx,numy)) #valores en 1 para saber que valores se extraen desde la imagen de entrenamiento
for i in xrange(numx*numy):
   TI[((BM[i][0] - minx)/tamx)][((BM[i][1] - miny)/tamy)] = BM[i][2]

print "       . . . Block model (training image) generated successfully "
print
# Obtaining training image finished

print "Opening and reading the simulation grid file (dimension):"
# it has to be like: simgrid.txt
grilla = np.loadtxt("simgrid_2d.txt")
sgnumx = int(grilla[0])
sgnumy = int(grilla[1])
sgtamx = int(grilla[2])
sgtamy = int(grilla[3])
sgminx = int(grilla[4])
sgminy = int(grilla[5])
SimGrid = -99.0 * np.ones((sgnumx,sgnumy, repeticion))
Pond = np.zeros((sgnumx,sgnumy, repeticion))
HData = -99.0 * np.ones((sgnumx,sgnumy, repeticion))
HData2 = -99.0 * np.ones(((sgnumx+2),(sgnumy+2), repeticion))
HData3 = -99.0 * np.ones(((sgnumx+2),(sgnumy+2), repeticion))
NonMoveData = np.zeros((sgnumx,sgnumy))
meanSG = np.zeros((sgnumx,sgnumy))
sgtotal =int((sgnumx*sgnumy))
print
print "Simulation grid created successfully"
print



print "Opening and reading the parameters file of the conditioning data: "
# it has to be like: parcd.txt
ParDataConditioning = np.loadtxt("parcd_2d.txt")
dcnumx = int(ParDataConditioning[0])
dcnumy = int(ParDataConditioning[1])
dctamx = int(ParDataConditioning[2])
dctamy = int(ParDataConditioning[3])
dcminx = int(ParDataConditioning[4])
dcminy = int(ParDataConditioning[5])
totaldc = int(ParDataConditioning[6])
print "       . . . conditioning data parameters loaded successfully "
print


print "Opening, reading and fixing the conditioning data file: "
# it has to be like cd.txt
ConDat=np.loadtxt("cd_2d.txt")
for ooo in xrange(repeticion):
   for var in xrange(totaldc):
      if sgminx <= ConDat[var][0] <= (sgminx + sgtamx*(sgnumx -1)): 
         if sgminy <=ConDat[var][1] <= (sgminy + sgtamy*(sgnumy-1)):
            aa= int((ConDat[var][0]-sgminx)/tamx)
            bb= int((ConDat[var][1]-sgminy)/tamy)
            SimGrid[aa][bb][ooo]=ConDat[var][2]
            Pond[aa][bb][ooo]= w1
            HData[aa][bb][ooo]=ConDat[var][2]
            NonMoveData[aa][bb]= 1



HData3[1:sgnumx+1,1:sgnumy+1,0:repeticion]= HData[0:sgnumx,0:sgnumy,0:repeticion]

promedio = 0
rr = 0
for enx in xrange(sgnumx):
   for eny in xrange(sgnumy):
      if HData[enx][eny][0] == -99.0:           
         for i in xrange(3):
            for j in xrange(3):
               if HData3[enx+i][eny+j][0] != -99.0:
                  promedio = promedio + HData3[enx+i][eny+j][0]
                  rr = rr + 1
               if rr>0:
                  promedio=promedio/rr
      if promedio >0:
         for ooo in xrange(repeticion):
            HData2[enx+1][eny+1][ooo] = promedio
         promedio = 0
         rr = 0

print "       . . . conditioning data loaded successfully "
print

print "Opening the average of each block"
promSG = np.loadtxt("mean_2d.txt")
print ".. done"
print 

# Aca se cargan los datos de la deriva en la media local
for i in xrange(sgnumx*sgnumy):
   meanSG[(promSG[i][0] - 1)][(promSG[i][1] - 1)]=promSG[i][2]



#Search's grid dimension
#print "Define the search grid size (must be odd values): "

# ngsx=int(raw_input("X axis :  "))
while ngsx<=1 or ngsx%2==0 or ngsx>sgnumx or ngsx>numx:
   print "Please put a value great than 1, odd, and less than the size"
   ngsx=int(raw_input("either the training image or the simulation grid:  "))

#print
# ngsy=int(raw_input("Y axis :  "))
while ngsy<=1 or ngsy%2==0 or ngsy>sgnumy or ngsy>numy:
   print "Please put a value great than 1, odd, and less than the size"
   ngsy=int(raw_input("either the training image or the simulation grid:  "))

#print
# ngsz=int(raw_input("Z axis :  "))
#while ngsz<=1 or ngsz%2==0 or ngsz>sgnumz or ngsz>numz:
#   print "Please put a value great than 1, odd, and less than the size"
#   ngsz=int(raw_input("either the training image or the simulation grid:  "))
print "Creating the filters"  
Fi = Filters(ngsx,ngsy) # Fi is a matrix wich include the total filters to be used
#print
print "... Filters created successfully"
#print

#Inner's grid dimension
#print "Define the inner grid size (must be odd values): "

# innerx=int(raw_input("X axis :  "))
while innerx<=1 or innerx%2==0 or innerx>ngsx or innerx>numx:
   print "Please put a value great than 1, odd, and less than the"
   innerx=int(raw_input("search grid size  "))

#print
# innery=int(raw_input("Y axis :  "))
while innery<=1 or innery%2==0 or innery>ngsy or innery>numy:
   print "Please put a value great than 1, odd, and less than the"
   innery=int(raw_input("search grid size:  "))

#print
# innerz=int(raw_input("Z axis :  "))
#while innerz<1 or innerz%2==0 or innerz>ngsz or innerz>numz:
#   print "Please put a value great than 1, odd, and less than the"
#   innerz=int(raw_input("search grid size:  "))

#print
#Here, the number of multi-grid is asked
# nmgrid=int(raw_input("Number of multigrid:  "))
while (nmgrid<=0) or ((2**(nmgrid-1))*ngsx>sgnumx and 2**(nmgrid-1)*ngsx>numx) or ((2**(nmgrid-1))*ngsy>sgnumy and 2**(nmgrid-1)*ngsy>numy):
   print "Please put a positive number, or small enough in order to be"   
   nmgrid=int(raw_input("within the simulated grid and training image: "))

par1 = int(numx-ngsx+1)
par2 = int(numy-ngsy+1)
print

camino = np.zeros((nmgrid,ncluster,2))
GSval = np.zeros((nmgrid,sgnumx,sgnumy)) # the grid will contain 1 in the position of the multigrid in order to cuantify the simulated nodes.
Listos = np.zeros((nmgrid,repeticion,sgnumx,sgnumy))
GSdatos = np.zeros((nmgrid,sgnumx,sgnumy))

patrones = -99.0*np.ones((nmgrid,par1,par2,8))
TImgrid=-99.0*np.ones((numx,numy,nmgrid))

#Here, the training image for each multigrid is processed

for xxxx in xrange(numx):
   for yyyy in xrange(numy):
      for ind in xrange(nmgrid):
         if (yyyy%2**(ind)==0) and (xxxx%2**(ind)==0):
            TImgrid[xxxx][yyyy][ind] = TI[xxxx][yyyy]
               
#Score map grid creation
maps=Scoremap(par1,par2,nmgrid)
depaso=np.zeros((ngsx,ngsy))
depaso2=np.zeros((ngsx,ngsy))

print "Now, creating the score maps"

z = 0
#These fors will go over the total scoremap getting the currently score 
for ind in xrange(nmgrid):
   for enx in xrange((numx+(2**(ind))*(-ngsx+1))):
      for eny in xrange((numy+(2**(ind))*(-ngsy+1))):
         for enxx in xrange(ngsx):
            for enyy in xrange(ngsy):
               depaso[enxx][enyy] = TI[enx+enxx*(2**ind)][eny+enyy*(2**ind)]

         maps.avtot[enx][eny][ind] = (depaso*Fi.avtot).sum()
         maps.avNS[enx][eny][ind] = (depaso*Fi.avNS).sum()
         maps.avEW[enx][eny][ind] = (depaso*Fi.avEW).sum()
         maps.curEW[enx][eny][ind] = (depaso*Fi.curEW).sum()
         maps.curNS[enx][eny][ind] = (depaso*Fi.curNS).sum()
         maps.gradEW[enx][eny][ind] = (depaso*Fi.gradEW).sum()
         maps.gradNS[enx][eny][ind] = (depaso*Fi.gradNS).sum()
         
         patrones[ind][enx][eny][7] = (depaso*Fi.avtot).sum()
         patrones[ind][enx][eny][0] = (depaso*Fi.avNS).sum()
         patrones[ind][enx][eny][1] = (depaso*Fi.avEW).sum()
         patrones[ind][enx][eny][2] = (depaso*Fi.curEW).sum()
         patrones[ind][enx][eny][3] = (depaso*Fi.curNS).sum()
         patrones[ind][enx][eny][5] = (depaso*Fi.gradNS).sum() 
         patrones[ind][enx][eny][6] = 0
         z=z+1

         
   print (ind+1), "over the", nmgrid, "scores maps done" 
print
print "       . . . score maps created successfully"
print


# Now, the implementation of the K-mean partition in order to obtain the pattern of each class
# The number of classes is variable, so it depends on the user.

for ind in xrange(nmgrid):
   pp1 = numx+(2**(ind))*(-ngsx+1)
   pp2 = numy+(2**(ind))*(-ngsy+1)
   tr = np.arange(pp1*pp2)
   np.random.shuffle(tr)   
   for i in xrange(ncluster):
      camino[ind][i][0]= int(tr[i]%pp1)
      camino[ind][i][1]= int((int(tr[i]/pp1))%pp2)
   tr = []

newcontador = np.zeros((nmgrid,ncluster))
print "Now, the clustering process is carried out"

Pattern = np.zeros((nmgrid,7,ncluster)) 
for ind in xrange(nmgrid):
   for var in xrange(ncluster):
      hi = camino[ind][var][0]
      hj = camino[ind][var][1]
      Pattern[ind][6][var] = maps.avtot[hi][hj][ind]
      Pattern[ind][0][var] = maps.avNS[hi][hj][ind]
      Pattern[ind][1][var] = maps.avEW[hi][hj][ind]
      Pattern[ind][2][var] = maps.curEW[hi][hj][ind]
      Pattern[ind][3][var] = maps.curNS[hi][hj][ind]
      Pattern[ind][4][var] = maps.gradEW[hi][hj][ind]
      Pattern[ind][5][var] = maps.gradNS[hi][hj][ind]

for uuu in xrange(kmeanveces):
   for ind in xrange(nmgrid):
      for enx in xrange((numx+(2**(ind))*(-ngsx+1))):
         for eny in xrange((numy+(2**(ind))*(-ngsy+1))):
            s0 = abs(patrones[ind][enx][eny][0] - Pattern[ind][0][0]) + abs(patrones[ind][enx][eny][1] - Pattern[ind][1][0]) + abs(patrones[ind][enx][eny][2] - Pattern[ind][2][0]) + abs(patrones[ind][enx][eny][3] - Pattern[ind][3][0]) + abs(patrones[ind][enx][eny][4] - Pattern[ind][4][0]) + abs(patrones[ind][enx][eny][5] - Pattern[ind][5][0]) #+ abs(patrones[enx][eny][enz][ind][9] - Pattern[ind][9][0])
            for ind2 in xrange(ncluster):
               s1 = abs(patrones[ind][enx][eny][0] - Pattern[ind][0][ind2]) + abs(patrones[ind][enx][eny][1] - Pattern[ind][1][ind2]) + abs(patrones[ind][enx][eny][2] - Pattern[ind][2][ind2]) + abs(patrones[ind][enx][eny][3] - Pattern[ind][3][ind2]) + abs(patrones[ind][enx][eny][4] - Pattern[ind][4][ind2]) + abs(patrones[ind][enx][eny][5] - Pattern[ind][5][ind2])   # + abs(patrones[enx][eny][enz][ind][9] - Pattern[ind][9][ind2])
               if s1<s0:
                  patrones[ind][enx][eny][6] = ind2
                  Pattern[ind][6][ind2] = ind2
                  s0 = abs(patrones[ind][enx][eny][0] - Pattern[ind][0][ind2]) + abs(patrones[ind][enx][eny][1] - Pattern[ind][1][ind2]) + abs(patrones[ind][enx][eny][2] - Pattern[ind][2][ind2]) + abs(patrones[ind][enx][eny][3] - Pattern[ind][3][ind2]) + abs(patrones[ind][enx][eny][4] - Pattern[ind][4][ind2]) + abs(patrones[ind][enx][eny][5] - Pattern[ind][5][ind2])# + abs(patrones[enx][eny][enz][ind][9] - Pattern[ind][9][ind2])
   
   
   for ind in xrange(nmgrid):
      for enx in xrange((numx+(2**(ind))*(-ngsx+1))):
         for eny in xrange((numy+(2**(ind))*(-ngsy+1))):
            newcontador[ind][int(patrones[ind][enx][eny][6])] += 1
            Pattern[ind][0][int(patrones[ind][enx][eny][6])] += patrones[ind][enx][eny][0]
            Pattern[ind][1][int(patrones[ind][enx][eny][6])] += patrones[ind][enx][eny][1]
            Pattern[ind][2][int(patrones[ind][enx][eny][6])] += patrones[ind][enx][eny][2]
            Pattern[ind][3][int(patrones[ind][enx][eny][6])] += patrones[ind][enx][eny][3]
            Pattern[ind][4][int(patrones[ind][enx][eny][6])] += patrones[ind][enx][eny][4]
            Pattern[ind][5][int(patrones[ind][enx][eny][6])] += patrones[ind][enx][eny][5]
            
   for ind in xrange(nmgrid):
      for ind2 in xrange(ncluster):
         if newcontador[ind][ind2] > 0:
            Pattern[ind][0][ind2] = (Pattern[ind][0][ind2])/(newcontador[ind][ind2])
   print "The amount of ", (uuu+1),"/",kmeanveces, "kmeans is done."

print "       . . . Clustering process done successfully"
print
print "el newcontador queda como:"
print newcontador

print "Now, the prototype of each class is carried out"


cont = np.zeros((nmgrid,ncluster)) # count the amount of pattern
cont2 = np.zeros((nmgrid,ncluster)) # count the amount of pattern
FinalPattern = np.zeros((nmgrid,ncluster,ngsx,ngsy)) 
y = (numx-ngsx+1)*(numy-ngsy+1)


f = 0
for ind in xrange(nmgrid):
   for enx in xrange((numx+(2**(ind))*(-ngsx+1))):
      for eny in xrange((numy+(2**(ind))*(-ngsy+1))):
         flecha = patrones[ind][enx][eny][6]
         cont[ind][flecha] = cont[ind][flecha] + 1
         for enxx in xrange(ngsx):
            for enyy in xrange(ngsy):
               depaso2[enxx][enyy] = TI[enx+enxx*(2**ind)][eny+enyy*(2**ind)]           
         FinalPattern[ind][flecha] = FinalPattern[ind][flecha] + depaso2
         f = f + 1
   f = 0
   for ind2 in xrange(ncluster):
      a = cont[ind][ind2]
      if a>0:
         FinalPattern[ind][ind2] = (1/a)*FinalPattern[ind][ind2]

print ".. making the dictionaries"
diccionario = {}
for ind in xrange(nmgrid):
   for ind2 in xrange(ncluster):
      if cont[ind][ind2]>0:
         diccionario[(ind,ind2)] = np.zeros(((cont[ind][ind2]),ngsx,ngsy))


for ind in xrange(nmgrid):
   for enx in xrange((numx+(2**(ind))*(-ngsx+1))):
      for eny in xrange((numy+(2**(ind))*(-ngsy+1))):
         flecha = patrones[ind][enx][eny][6]
         depaso3 = diccionario[(ind,flecha)]
         fle = cont2[ind][flecha]
         for enxx in xrange(ngsx):
            for enyy in xrange(ngsy):
               depaso3[fle][enxx][enyy] = TI[enx+enxx*(2**ind)][eny+enyy*(2**ind)]

         diccionario[(ind,flecha)] = depaso3
         cont2[ind][flecha] = cont2[ind][flecha] + 1


print "       . . . Prototype created successfully"
print


# ########################################################################################
# ########################################################################################
# ######################## SIMULATION PROCESS ############################################
# ########################################################################################
# ########################################################################################

#Now, the simulation process is carried out

print "Now, the simulation process is carried out"

for trab in xrange(repeticion):
   
   GSind = -99.0 * np.ones((nmgrid,sgnumx,sgnumy))# the grid will contain the random number of the random path whitin the "imagen erodada"
   print (trab+1), "over the", repeticion, "simulations is carried out"
  
   if nmgrid == 1 :
      ind = 0
      t=(sgnumx+(-ngsx+1))*(sgnumy+(-ngsy+1))
      ara = np.arange(t) 
      np.random.shuffle(ara)
      h = 0
      dondex = -99.0*np.ones(t)
      dondey = -99.0*np.ones(t)
      #dondez = -99.0*np.ones(t)
      xx = (ngsx-1)/2
      yy = (ngsy-1)/2
      #zz = (ngsz-1)/2 
      
      xx2 = (ngsx-innerx)/2
      yy2 = (ngsy-innery)/2
      #zz2 = (ngsz-innerz)/2    
      
      xx3 = (innerx-1)/2
      yy3 = (innery-1)/2
      #zz3 = (innerz-1)/2       
      vuelta = 1
      
      # Here the GSind will indicate the random path over the simulation grid
      for enx in xrange(sgnumx-(ngsx-1)):
         for eny in xrange(sgnumy-(ngsy-1)):
            #for enz in xrange(sgnumz-(ngsz-1)):
            GSind[ind][enx + (xx)][eny + (yy)] = ara[h]
            dondex[ara[h]] = (enx + (xx))
            dondey[ara[h]] = (eny + (yy))
            #dondez[ara[h]] = (enz + (zz))
            h=h+1
       
      GSdatos[ind,xx2:sgnumx-(ngsx-innerx) +xx2,yy2:sgnumy-(ngsy-innery)+yy2] = 1
      
      com = GSdatos[ind].sum()  
      w = 9999999999999999999999999.0
      while (com>0):
         for d in xrange(t): # Run all over each point
            pos = 0
            cntd = 0
            cntd2 = 0
            menx = int(dondex[d])
            meny = int(dondey[d])
            #menz = int(dondez[d])
            com1 = 0
            if Listos[ind][trab][menx][meny] == 0:
               com1=GSdatos[ind,menx-xx:menx-xx+ngsx,meny-yy:meny-yy+ngsy].sum()
            if com1 == 0:
               Listos[ind][trab][menx][meny] = 1 # 1: full simulated; 0: not-full simulated
            w = 9999999999999999999999999.0
            com = GSdatos.sum()
            if com == 0:
               break
        
            if (com1 >0): #and Aval[ind][menx][meny][menz]==1:              
               #Here the process  of compare each prototype is carried out
               #This for count the amount of hard data whithin the search grid
               cntd = NonMoveData[menx-xx:menx-xx+ngsx,meny-yy:meny-yy+ngsy].sum()
                                                 
               if cntd>=minimumdc and vuelta == 1:
                  for ind4 in xrange(ncluster):
                     p = 0.0
                     if cont[ind][ind4]>0:
                        
                        p1 = Pond[menx-xx:menx-xx+ngsx,meny-yy:meny-yy+ngsy,trab]*(abs(FinalPattern[ind][ind4]-SimGrid[menx-xx:menx-xx+ngsx,meny-yy:meny-yy+ngsy,trab]))
                        p1 = p1.sum()
                        promActual=np.mean(FinalPattern[ind][ind4])
                        p=p1*(1-w0)+w0*(abs(meanSG[menx][meny] - promActual))
                              
                        if p<w:
                           w = p
                           pos = ind4
                        # Now, the position "pos" indicate the closest prototype to the conditioning event
                        # a random pattern has to be chosen within the prototype class                 
                  eleccion = int(cont[ind][pos]*ran.random())
                  depaso4 = diccionario[(ind,pos)]      
                  #Here the paste of the data is carried out
                  SimGrid[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab] = depaso4[eleccion]               
                  SimGrid[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab] = np.where(NonMoveData[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery]<1,SimGrid[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab],HData[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab])

                  GSdatos[ind,menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery] = 0
                  Pond[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab] = np.where(NonMoveData[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery]>0,Pond[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab],w2)
               if vuelta>1:
                  vacio = Pond[menx-xx:menx-xx+ngsx,meny-yy:meny-yy+ngsy,trab].sum()
                  if vacio>0:
                     for ind4 in xrange(ncluster):
                        p = 0.0
                        if cont[ind][ind4]>0:
                           p1 = Pond[menx-xx:menx-xx+ngsx,meny-yy:meny-yy+ngsy,trab]*(abs(FinalPattern[ind][ind4]-SimGrid[menx-xx:menx-xx+ngsx,meny-yy:meny-yy+ngsy,trab]))
                           p1 = p1.sum()
                           promActual=np.mean(FinalPattern[ind][ind4])
                           p=p1*(1-w0)+w0*(abs(meanSG[menx][meny] - promActual))                        
                        
                           if p<w:
                              w = p
                              pos = ind4
                           # Now, the position "pos" indicate the closest prototype to the conditioning event
                           # a random pattern has to be chosen within the prototype class
                  if vacio == 0:
                     tt = 0
                     while(tt<1):
                        pos = int(ran.random()*ncluster)
                        if diccionario.has_key((ind,pos)):
                           tt=1                     
                     
                  eleccion = int(cont[ind][pos]*ran.random()) 
                  depaso4 = diccionario[(ind,pos)]      
                  SimGrid[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab] = depaso4[eleccion] #[xx2:(ngsx-xx2),yy2:(ngsy-yy2)]
                  SimGrid[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab] = np.where(NonMoveData[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery]<1,SimGrid[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab],HData[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab])
                  GSdatos[ind,menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery] = 0
                  Pond[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab] = np.where(NonMoveData[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery]>0,Pond[menx-xx3:menx-xx3+innerx,meny-yy3:meny-yy3+innery,trab],w2)
         com = GSdatos.sum()
         com1 = 0
         vuelta = vuelta + 1 


   # Here the "simulation process when the multigrid is more than 1"  

   if nmgrid != 1 :
      for ind3 in xrange(nmgrid):
         print "...", (ind3+1), "over the", (nmgrid), "multigrid is carried out"
         ind = nmgrid-ind3-1 # this index "ind" is made for count from the maximum to the minimum multi grid      
         t=(sgnumx+(2**(ind))*(-ngsx+1))*(sgnumy+(2**(ind))*(-ngsy+1))     
         ara = np.arange(t) 
         np.random.shuffle(ara)
         h = 0
         dondex = -99.0*np.ones(t)
         dondey = -99.0*np.ones(t)
         #dondez = -99.0*np.ones(t)
         xx = (2**(ind))*(ngsx-1)/2 
         yy = (2**(ind))*(ngsy-1)/2
         #zz = (2**(ind))*(ngsz-1)/2 
         xx2 = (2**(ind))*(ngsx-innerx)/2 
         yy2 = (2**(ind))*(ngsy-innery)/2 
         #zz2 = (2**(ind))*(ngsz-innerz)/2   
         xx3 = (2**(ind))*(innerx-1)/2 
         yy3 = (2**(ind))*(innery-1)/2
         #zz3 = (2**(ind))*(innerz-1)/2
        
         vuelta = 1
         # Here the GSind will indicate the random path over the simulation grid
         for enx in xrange(sgnumx+(2**(ind))*(-ngsx+1)): #
            for eny in xrange(sgnumy+(2**(ind))*(-ngsy+1)): #(2**(ind))*(-ngsy+1)
               #for enz in xrange(sgnumz+(2**(ind))*(-ngsz+1)):
               if ((enx+xx)%(2**(ind))==0) and ((eny+yy)%(2**(ind))==0):
                  GSind[ind][enx + (xx)][eny + (yy)] = ara[h]
                  dondex[ara[h]] = (enx + (xx))
                  dondey[ara[h]] = (eny + (yy))
                  #dondez[ara[h]] = (enz + (zz))
                  #GSdatos[ind][enx + (xx)][eny + (yy)] = 1
                  Listos[ind][trab][enx + (xx)][eny + (yy)] = 1
                  h=h+1
         for enx in xrange(sgnumx+(2**(ind))*(-ngsx+innerx)): #
            for eny in xrange(sgnumy+(2**(ind))*(-ngsy+innery)): #(2**(ind))*(-ngsy+1)
               #for enz in xrange(sgnumz+(2**(ind))*(-ngsz+1)):
               if ((enx+xx2)%2**(ind)==0) and ((eny+yy2)%2**(ind)==0):   
                  GSdatos[ind][enx + (xx2)][eny + (yy2)] = 1
                  GSval[ind][enx + (xx2)][eny + (yy2)] = 1    
         com = Listos[ind][trab].sum() 
         
         w = 9999999999999999999999999.0
         
         while (com>0):
            for d in xrange(t): # Run all over each point
               pos = 0
               cntd = 0
               cntd2 = 0
               menx = int(dondex[d])
               meny = int(dondey[d])
               #menz = int(dondez[d])
               com1 = 0
                           
               if menx != -99 and meny != -99:
                  #print ind, menx, meny, trab
                  if Listos[ind][trab][menx][meny] == 1: 
                  #if GSdatos[ind][menx][meny] == 1: #
                     
                     com1=GSdatos[ind,menx-xx3:menx+xx3,meny-yy3:meny+yy3].sum()
                     #print "com1 es:", com1
                     
                     # If you want to use a DUAL GRID display the code below
                     '''
                     if ind==0: # 
                       #print "por aqui"
                        for posi in xrange(3):
                           for posj in xrange(3):
                              #for posk in xrange(3):
                              if NonMoveData[menx+posi-1][meny+posj-1] != 1: 
                                 try:
                                    GSval[ind+1][menx+posi-1][meny+posj-1]==0
                                    SimGrid[menx+posi-1][meny+posj-1][trab] = np.mean(SimGrid[menx+posi-2:menx+posi,meny+posj-2:meny+posj,trab]*GSval[ind+1,menx+posi-2:menx+posi,meny+posj-2:meny+posj])
                                    Pond[menx+posi-1][meny+posj-1][trab] = w3
                                 except IndexError:
                                    continue
                        '''     
               
                     if com1 == 0:
                        Listos[ind][trab][menx][meny] = 0 # 1: full simulated; 0: not-full simulated   for enx in xrange(ngsx):

                     
                     
               else:
                  com1=0
               com = Listos[ind][trab].sum() 
               if com == 0:
                  break               
               w = 9999999999999999999999999.0
               if (com1 >0): 
                  #Here the process  of compare each prototype is carried out
                  #This for count the amount of hard data whithin the search grid
                  contador = GSval[ind,menx-xx:menx+xx,meny-yy:meny+yy]*NonMoveData[menx-xx:menx+xx,meny-yy:meny+yy]
                  cntd = contador.sum()

                  if cntd>=minimumdc and vuelta == 1:
                     for ind4 in xrange(ncluster):
                        p = 0.0
                        
                        if cont[ind][ind4]>0:
                           p1=0
                           for i1 in xrange(ngsx):
                              for j1 in xrange(ngsy):
                                 p1=p1 + GSval[ind][menx+(2**(ind))*i1-xx][meny+(2**(ind))*j1-yy]*(Pond[menx+(2**(ind))*i1-xx][meny+(2**(ind))*j1-yy][trab]*(abs(FinalPattern[ind][ind4][i1][j1] - SimGrid[menx+(2**(ind))*i1-xx][meny+(2**(ind))*j1-yy][trab])))
                           
                           p1 = p1.sum()
                           
                           promActual=np.mean(FinalPattern[ind][ind4])
                           p=p1*(1-w0)+w0*(abs(meanSG[menx][meny] - promActual))
                           
                                
                           if p<w:
                              w = p
                              pos = ind4
                           # Now, the position "pos" indicate the closest prototype to the conditioning event
                           # a random pattern has to be chosen within the prototype class
   
   
                     eleccion = int(cont[ind][pos]*ran.random())
                     depaso4 = diccionario[(ind,pos)]      
                     #Here the paste of the data is carried out
                     # Then "li" fix the position in the array that gives the position in the TI.  
                                
                     for i in xrange(innerx):
                        for j in xrange(innery):         
                           ssx = i + (ngsx-1)/2 - ((innerx-1)/2)                       
                           ssy = j + (ngsy-1)/2 - ((innery-1)/2)
                           if NonMoveData[(2**(ind))*i + menx - (2**(ind))*((innerx-1)/2)][(2**(ind))*j + meny - (2**(ind))*((innery-1)/2)] != 1: # Here, each conditional data is not repleaced 
                              SimGrid[(2**(ind))*i + menx - (2**(ind))*((innerx-1)/2)][(2**(ind))*j + meny - (2**(ind))*((innery-1)/2)][trab] = depaso4[eleccion][ssx][ssy]
                              Pond[(2**(ind))*i + menx - (2**(ind))*((innerx-1)/2)][(2**(ind))*j + meny - (2**(ind))*((innery-1)/2)][trab] = w2
   
                     #Here, the locations of the values inside the inner grid are realesed             
                     GSdatos[ind,menx-xx3:menx+xx3,meny-yy3:meny+yy3]=0
                     Listos[ind][trab][menx][meny] = 0
                     dondex[GSind[ind][menx][meny]] = -99.0
                     dondey[GSind[ind][menx][meny]] = -99.0
                  if vuelta>1:
                     #check there are not empty space within the grid 
                     vacio = Pond[menx-xx:menx+xx,meny-yy:meny+yy,trab].sum()
                     if vacio>0:
                        for ind4 in xrange(ncluster):
                           p = 0.0
                           if cont[ind][ind4]>0:
                              p1=0
                              for i1 in xrange(ngsx):
                                 for j1 in xrange(ngsy):
                                    p1=p1 + GSval[ind][menx+(2**(ind))*i1-xx][meny+(2**(ind))*j1-yy]*(Pond[menx+(2**(ind))*i1-xx][meny+(2**(ind))*j1-yy][trab]*(abs(FinalPattern[ind][ind4][i1][j1] - SimGrid[menx+(2**(ind))*i1-xx][meny+(2**(ind))*j1-yy][trab])))
                              
                              p1 = p1.sum()
                              promActual=np.mean(FinalPattern[ind][ind4])
                              p=p1*(1-w0)+w0*(abs(meanSG[menx][meny] - promActual))                           
                              if p<w:
                                 w = p
                                 pos = ind4
                              # Now, the position "pos" indicate the closest prototype to the conditioning event
                              # a random pattern has to be chosen within the prototype class
      
                     if vacio ==0:
                        tt = 0
                        while(tt<1):
                           pos = int(ran.random()*ncluster)
                           if diccionario.has_key((ind,pos)):
                              tt=1
                        
                     eleccion = int(cont[ind][pos]*ran.random()) 
                     depaso4 = diccionario[(ind,pos)]      
                     #Here the paste of the data is carried out
                     for i in xrange(innerx):
                        for j in xrange(innery):

                           ssx = i + (ngsx-1)/2 - ((innerx-1)/2)                       
                           ssy = j + (ngsy-1)/2 - ((innery-1)/2)
                           if NonMoveData[(2**(ind))*i + menx - (2**(ind))*((innerx-1)/2)][(2**(ind))*j + meny - (2**(ind))*((innery-1)/2)] != 1: # Here, each conditional data is not repleaced 
                              SimGrid[(2**(ind))*i + menx - (2**(ind))*((innerx-1)/2)][(2**(ind))*j + meny - (2**(ind))*((innery-1)/2)][trab] = depaso4[eleccion][ssx][ssy]
                              Pond[(2**(ind))*i + menx - (2**(ind))*((innerx-1)/2)][(2**(ind))*j + meny - (2**(ind))*((innery-1)/2)][trab] = w2

                     GSdatos[ind,menx-xx3:menx+xx3,meny-yy3:meny+yy3]=0
                     Listos[ind][trab][menx][meny] = 0
                     dondex[GSind[ind][menx][meny]] = -99.0
                     dondey[GSind[ind][menx][meny]] = -99.0
                     
            com = Listos[ind][trab].sum() 
            com1=0     
            vuelta = vuelta + 1 
         if ind>0:
            Pond[0:sgnumx,0:sgnumy,trab]=np.where(Pond[0:sgnumx,0:sgnumy,trab]!=w2,Pond[0:sgnumx,0:sgnumy,trab],w3)

   print (trab+1), "over the", repeticion, "simulations done." 
   print
  
# ########################################################################################
# ########################################################################################
# ########################################################################################

print "Saving result's files"

# This section save as out file the total scores maps

d = 0 
'''
with open('Scoremaps_2d.txt', 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter = '\t')
   spamwriter.writerow(["Scoremaps_2d"])
   cantidad = 2 + 7*nmgrid
   spamwriter.writerow([cantidad])
   spamwriter.writerow(["X"])
   spamwriter.writerow(["Y"])
   #spamwriter.writerow(["Z"])
   for ele in xrange(nmgrid):
      #spamwriter.writerow(["AvTotal_%.0f" %(ele+1)])
      #spamwriter.writerow(["AvELE_%.0f" %(ele+1)])
      spamwriter.writerow(["AvEW_%.0f" %(ele+1)])
      spamwriter.writerow(["AvNS_%.0f" %(ele+1)])
      #spamwriter.writerow(["CurELE_%.0f" %(ele+1)])
      spamwriter.writerow(["CurEW_%.0f" %(ele+1)])
      spamwriter.writerow(["CurNS_%.0f" %(ele+1)])
      #spamwriter.writerow(["GradELE_%.0f" %(ele+1)])
      spamwriter.writerow(["GradEW_%.0f" %(ele+1)])
      spamwriter.writerow(["GradNS_%.0f" %(ele+1)])
      spamwriter.writerow(["Prototype_%.0f" %(ele+1)])
   
   for xxx in xrange(par1):
      for yyy in xrange(par2):
         #for zzz in xrange(par3):
         d = [xxx] + [yyy] + [zzz]
         for ind in xrange(nmgrid):
            #spamwriter = csv.writer(csvfile, delimiter = ',')
            #av0 = maps.avtot[xxx][yyy][zzz][ind]
            #av1 = maps.avELE[xxx][yyy][zzz][ind]
            av2 = maps.avEW[xxx][yyy][ind]
            av3 = maps.avNS[xxx][yyy][ind]
            #cur1 = maps.curELE[xxx][yyy][zzz][ind]
            cur2 = maps.curEW[xxx][yyy][ind]
            cur3 = maps.curNS[xxx][yyy][ind]
            #grad1 = maps.gradELE[xxx][yyy][zzz][ind]
            grad2 = maps.gradEW[xxx][yyy][ind]
            grad3 = maps.gradNS[xxx][yyy][ind]
            prot = patrones[xxx][yyy][ind][6]
            d = d + [str(av2)]+[str(av3)]+[str(cur2)]+[str(cur3)]+[str(grad3)]+[str(grad3)]+[str(prot)]
         spamwriter.writerow(d)
         d=[]
# Here the output file is finished


with open('FinalPrototype.txt', 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter = '\t')
   spamwriter.writerow(["FinalPrototype"])
   spamwriter.writerow(["5"])
   spamwriter.writerow(["Multigrid"])
   spamwriter.writerow(["Ncluster"])
   spamwriter.writerow(["X"])
   spamwriter.writerow(["Y"])
   #spamwriter.writerow(["Z"])
   spamwriter.writerow(["Value"])
   for ind in xrange(nmgrid):
      for ind3 in xrange(ncluster):
         d = [ind] + [ind3]
         for xxx in xrange(ngsx):
            for yyy in xrange(ngsy):
               #for zzz in xrange(ngsz):
               d = d + [xxx] + [yyy]
               ble = FinalPattern[ind][ind3][xxx][yyy]
               d = d + [str(ble)]
               spamwriter.writerow(d)
               d = [ind] + [ind3]
      d=[]                  
               
'''
# This section save as out file the total simulation grid
with open('MainResults.txt', 'wb') as csvfile:
   elementos = repeticion+3
   spamwriter = csv.writer(csvfile, delimiter = '\t')
   spamwriter.writerow(["Results"])
   spamwriter.writerow([elementos])
   spamwriter.writerow(["X"])
   spamwriter.writerow(["Y"])
   #spamwriter.writerow(["Z"])
   for dale in xrange(repeticion):
      c = ["Grade %.0f" %(dale+1)]
      spamwriter.writerow(c)
   spamwriter.writerow(["Average"])
   
   for xxx in xrange(sgnumx):
      for yyy in xrange(sgnumy):
         #for zzz in xrange(sgnumz):
         xax = (sgminx +sgtamx*(xxx))
         yay = (sgminy +sgtamy*(yyy))
         #zaz = (sgminz +sgtamz*(zzz))
         b=[xax] + [yay] 
         g = 0
         iii = 0
         prom = 0
         for can in xrange(repeticion):
            lo = SimGrid[xxx][yyy][can]
            if lo >= 0:
               g = g + lo
               iii = iii + 1
            b = b + [str(SimGrid[xxx][yyy][can])]
         if iii >0:
            prom = g/repeticion
         b = b + [str(prom)]
         spamwriter.writerow(b)
         b=[]
         prom = 0
            
# Here the output file is finished
'''
# This section save as out file the total simulation grid
with open('results.txt', 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter = '\t')
  
   for xxx in xrange(sgnumx):
      for yyy in xrange(sgnumy):
         for zzz in xrange(sgnumz):
            xax = (sgminx +sgtamx*(xxx))
            yay = (sgminy +sgtamy*(yyy))
            zaz = (sgminz +sgtamz*(zzz))
            b=[xax] + [yay] + [zaz]
            g = 0
            prom = 0
            for can in xrange(repeticion):
               lo = SimGrid[xxx][yyy][zzz][can]
               if lo >= 0:
                  g = g + lo

               b = b + [str(SimGrid[xxx][yyy][zzz][can])]

            spamwriter.writerow(b)
            b=[]
            prom = 0
'''
# Here the output file is finished

print "End of the simulation. Thanks."


