# -*- coding: utf-8 -*-
"""
DocString
"""

# Import modules and set-up Matplotlib
import matplotlib
from matplotlib import rc
matplotlib.use('Agg')
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.cm
import fipy
import numpy
import scipy.special
import time
import multiprocessing
import os
import os.path



######################### User Specified Parameters #########################

# Set font used in plots (Global)
Font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

# Color scheme to be used in plots. (Global)
PlotColor = matplotlib.cm.brg

# When smaller time steps are desired, the default time step may be
# sub-divided. Sets sub-divisions per time step. Should always have 
# SubDiv >= 1.0 (Global)
SubDiv = 1.0

# Diffusion time scale (Global)
Tau = 1.0


# Number of divisions per thermal length in phase space
n1 = 5

# Number of thermal lengths in phase space domain
n2 = 5

# Set to "True" if distributions, transfer functions, and plots 
# should be saved
saveResults = True

# Directory path to where data should be saved
# This should point to an empty directory
pathStr = '/scratch/sigurd/Static-1/'

# Initial time for evolution
tStart = 0

# Final time for evolution
tFinal = 1.

# Result of measurement at end of experiment
mX = 2.
mP = None

# Width of Gaussian measurement
sigmaX = .2
sigmaP = None

# Define spring constant as a function of time
def springConstant(t):
    # Spring constant should always be equal to one at
    # the initial time t=tStart
    k = 1.
    return k

# Define minimum of potential as a function of time
def potentialMinimum(t):
    # The potential's minimum should always be at 
    # zero at the initial time t=tStart
    z = 0.
    return z

# Define initial distribution
def distributionInitial():
    phi = fipy.CellVariable(mesh=Mesh, name=r"$\rho(x,p)$")
    X, P = Mesh.cellCenters()
    temp = numpy.exp(-(1./2)*(X)**2-(1./2)*(P)**2)
    phi.setValue(temp)
    norm = fipy.numerix.sum(phi.value, axis=0)*Dx*Dp
    phi.setValue(phi.value/norm)
    del temp, norm, X, P
    return phi

#############################################################################
######################### DO NOT EDIT BELOW HERE ############################
#############################################################################

############################### Functions ###################################

# Wrapper for evolving a distribution
def evolution(phi, tStart, tEnd, path, saveResults):
    if(tStart<=tEnd):
        phi = forwardEvo(phi, tStart, tEnd, path, saveResults)
    else:
        phi = reverseEvo(phi, tStart, tEnd, path, saveResults)
    return phi


# Evolves a distribution under the forward-time protocol
def forwardEvo(phi, tStart, tEnd, path, saveResults):
    # Initialize step counter and evolution time
    step = 0
    tEvo = tStart
    # Extract cell centers and faces from mesh
    X, P = Mesh.cellCenters()
    xFace, pFace = Mesh.faceCenters()
    # Create plot for starting distribution
    target = 'forwardEvo-Step-' + '%05d'%step + '.png'
    genPlots(phi, tEvo, saveResults, target, path)
    # Continue to evolve distribution until specified end time
    while(tEvo<tEnd):
        # Find current spring constant and location of minimum
        kT = springConstant(tEvo)
        zT = potentialMinimum(tEvo)
        # Calculate time step size
        timeStepDuration=(1./(SubDiv*Nx))*(1./numpy.sqrt(1./Tau**2+kT))
        if(tEvo+timeStepDuration>tEnd):
            timeStepDuration = tEnd - tEvo
        # Create Diffusion Term
        gxx = numpy.zeros(X.shape)
        gxp = numpy.zeros(X.shape)
        gpx = numpy.zeros(X.shape)
        gpp = numpy.ones(X.shape)*(2./Tau)
        dTerm = fipy.DiffusionTerm(fipy.CellVariable(mesh=Mesh, 
                                            value=[[gxx,gxp],[gpx,gpp]]))
        del gxx, gxp, gpx, gpp
        # Create convection term
        uX = pFace
        uP = -kT*(xFace-zT) - (2./Tau)*pFace
        cCoeff = fipy.FaceVariable(mesh=Mesh, value=[uX,uP])
        cTerm = fipy.ExponentialConvectionTerm(cCoeff)
        del uX, uP, cCoeff
        # Create evolution equation
        eq = fipy.TransientTerm()+cTerm==dTerm
        # Specify solver
        solver = fipy.solvers.pysparse.LinearLUSolver(tolerance=10**-15, \
                                                iterations=1000, precon=None)
        # Evolve system
        eq.solve(var=phi, dt=timeStepDuration, solver=solver)
        tEvo = tEvo + timeStepDuration
        step = step + 1
        # Check normalization for possible errors
        norm = fipy.numerix.sum(phi.value, axis=0)*Dx*Dp
        if(abs(norm-1)>10**(-13)):
            s1 = 'Distribution is no longer normalized.\n'
            s2 = 'Abs(1-Norm) = '
            s3 = repr(norm-1.)
            raise RuntimeError(s1+s2+s3)
        del kT, zT, timeStepDuration, cTerm, eq, norm
        # Create plot of current distribution
        target = 'forwardEvo-Step-' + '%05d'%step + '.png'
        genPlots(phi, tEvo, saveResults, target, path)
    del tEvo, X, P, xFace, pFace
    return phi


# Evolves a distribution under the time-reversed protocol
def reverseEvo(phi, tStart, tEnd, path, saveResults):
    # Initialize step counter and evolution time
    step = 0
    tEvo = tStart
    # Extract cell centers and faces from mesh
    X, P = Mesh.cellCenters()
    xFace, pFace = Mesh.faceCenters()
    # Create plot for starting distribution
    if(saveResults):
        # Plot conjugate of current distribution
        prob = phi.value
        conjProb = conjugateDist(prob)
        phiConj = fipy.CellVariable(mesh=Mesh, name=r"\LARGE $\rho(x,p)$")
        phiConj.setValue(conjProb)
        del conjProb, prob
        target = 'reverseEvo-Step-' + '%05d'%step + '.png'
        genPlots(phiConj, tEvo, saveResults, target, path)
        del phiConj
    # Continue to evolve distribution until specified end time
    while(tEvo>tEnd):
        # Find current spring constant and location of minimum
        kT = springConstant(tEvo)
        zT = potentialMinimum(tEvo)
        # Calculate time step size
        timeStepDuration=(1./(SubDiv*Nx))*(1./numpy.sqrt(1./Tau**2+kT))
        if(tEvo-timeStepDuration<tEnd):
            timeStepDuration = tEvo - tEnd
        # Create Diffusion Term
        gxx = numpy.zeros(X.shape)
        gxp = numpy.zeros(X.shape)
        gpx = numpy.zeros(X.shape)
        gpp = numpy.ones(X.shape)*(2./Tau)
        dTerm = fipy.DiffusionTerm(fipy.CellVariable(mesh=Mesh, 
                                            value=[[gxx,gxp],[gpx,gpp]]))
        del gxx, gxp, gpx, gpp
        # Create convection term
        uX = pFace
        uP = -kT*(xFace-zT) - (2./Tau)*pFace
        cCoeff = fipy.FaceVariable(mesh=Mesh, value=[uX,uP])
        cTerm = fipy.ExponentialConvectionTerm(cCoeff)
        del uX, uP, cCoeff
        # Create evolution equation
        eq = fipy.TransientTerm()+cTerm==dTerm
        # Specify Solver
        solver = fipy.solvers.pysparse.LinearLUSolver(tolerance=10**-15, \
                                                iterations=1000, precon=None)
        # Evolve system
        eq.solve(var=phi, dt=timeStepDuration, solver=solver)
        tEvo = tEvo - timeStepDuration
        step = step + 1
        # Check normalization for possible errors
        norm = fipy.numerix.sum(phi.value, axis=0)*Dx*Dp
        if(abs(norm-1)>10**(-13)):
            s1 = 'Distribution is no longer normalized.\n'
            s2 = 'Abs(1-Norm) = '
            s3 = repr(norm-1.)
            raise RuntimeError(s1+s2+s3)
        del kT, zT, timeStepDuration, cTerm, eq, norm
        # Create plot of current distribution
        if(saveResults):
            # Plot conjugate of current distribution
            prob = phi.value
            conjProb = conjugateDist(prob)
            phiConj = fipy.CellVariable(mesh=Mesh, name=r"\LARGE $\rho(x,p)$")
            phiConj.setValue(conjProb)
            del conjProb, prob
            target = 'reverseEvo-Step-' + '%05d'%step + '.png'
            genPlots(phiConj, tEvo, saveResults, target, path)
            del phiConj            
    del tEvo, X, P, xFace, pFace
    return phi


# Update the initial distribution using the impulse response functions
def updateInitial(unupdatedValues, responseFunctions, mx, mp, \
                                                            sigmaX, sigmaP):
    # Extract cell centers from mesh
    X = Mesh.x()
    P = Mesh.y()
    # Find array of measurement probabilities
    measProbs = measurement(mx, mp, X, P, sigmaX, sigmaP)
    measProbs = measProbs.reshape((1,measProbs.size))
    # Update initial distribution
    unnormedUpdate = unupdatedValues*Dx*Dp*\
                        numpy.sum(responseFunctions*measProbs, axis=1)
    norm = numpy.sum(unnormedUpdate)*Dx*Dp
    updatedValues = unnormedUpdate/norm
    del unnormedUpdate, norm
    return updatedValues
    

# Update a distribution at the measurement time based on the outcome
def updateDistribution(unupdatedValues, mx, mp, sigmaX, sigmaP):
    # Extract cell centers
    X = Mesh.x()
    P = Mesh.y()
    # Find measurement probabilities
    measProb = measurement(mx, mp, X, P, sigmaX, sigmaP)
    # Update distribution
    unnormedUpdate = measProb*unupdatedValues
    norm = numpy.sum(unnormedUpdate)*Dx*Dp
    updatedValues = unnormedUpdate/norm
    del measProb, unnormedUpdate, norm
    return updatedValues


# For a given a microstate (Sx, Sp), returns the probability of measuring 
# (Mx, Mp), i.e. P(m|x). Accepts Sx and Sp as numpy arrays of equal length.
def measurement(mx, mp, Sx, Sp, sigmaX, sigmaP):
    # Find how many microstates are being requested
    nStates = Sx.size
    # Extract cell centers
    X, P = Mesh.cellCenters()
    # Reshape arrays
    Xshaped = X.reshape((1,X.size))
    Pshaped = P.reshape((1,P.size))
    Sxshaped = Sx.reshape((Sx.size,1))
    Spshaped = Sp.reshape((Sp.size,1))
    # Vectorize calculation
    Xtiled = numpy.tile(Xshaped, (nStates,1))
    Ptiled = numpy.tile(Pshaped, (nStates,1))
    Sxtiled = numpy.tile(Sxshaped, (1, X.size))
    Sptiled = numpy.tile(Spshaped, (1, P.size))
    # Calculate distribution of measuring device
    if((sigmaX==None)&(sigmaP==None)):
        temp = numpy.ones(Xtiled.shape)
    elif((sigmaX==None)):
        temp = numpy.exp(-((Sptiled-Ptiled)/sigmaP)**2/2)
    elif((sigmaP==None)):
        temp = numpy.exp(-((Sxtiled-Xtiled)/sigmaX)**2/2)
    else:
        temp = numpy.exp(-((Sxtiled-Xtiled)/sigmaX)**2/2 \
                                -((Sptiled-Ptiled)/sigmaP)**2/2)
    norms = numpy.sum(temp, axis=1)*Dx*Dp
    norms = norms.reshape((norms.size, 1))
    measDist = temp/norms
    # Find cell center(s) closest to the specified measurement
    if((mx==None)&(mp==None)):
        cellIDs = numpy.unique(Mesh._getNearestCellID(([X],[P])))
    elif((mx==None)):
        cellIDs = numpy.unique(Mesh._getNearestCellID(([X],[mp])))
    elif((mp==None)):
        cellIDs = numpy.unique(Mesh._getNearestCellID(([mx],[P])))
    else:
        cellIDs = numpy.unique(Mesh._getNearestCellID(([mx],[mp])))
    # Find total probability of measurement
    measProbs = measDist[:,cellIDs]
    measProb = numpy.sum(measProbs, axis=1)
    measProb = measProb.reshape((measProb.size,))
    del temp, norms, X, P, measDist
    return measProb


# Calculates impulse response functions from initial time
def impulseResponse(argArray):
    # Extract arguments
    tStart = argArray[0]
    tFinal = argArray[1]
    k = argArray[2]
    # Initialize the impulse distribution
    phi = fipy.CellVariable(mesh=Mesh, name=r"$\rho(x,p)$")
    temp = numpy.zeros(((2*Nx)**2,))
    temp[k]=1.
    norm = Dx*Dp
    phi.setValue(temp/norm)
    # Evolve the distribution
    response = evolution(phi, tStart, tFinal, '', False)
    return response


# Calculates te relative entropy of two distributions
def relEntropy(phi1, phi2):
    v1 = phi1.value
    v2 = phi2.value
    sTemp = scipy.special.xlogy(v1, v1/v2)*Dx*Dp
    # Remove spurious NaN's that appear due to 0/0
    sTemp[v1==0] = 0
    sRelative = numpy.sum(sTemp)
    del v1, v2, sTemp
    return sRelative
    

# Create and display plots of a distribution    
def genPlots(phi, t, saveResults, targetName, path):
    if(saveResults):
        titleStr = 't = ' + '%.5f' % t
        viewerTemp = fipy.Matplotlib2DViewer(vars=phi, title=titleStr, \
                        limits={'xmin':-XMax, 'xmax':XMax, 
                        'ymin':-PMax, 'ymax':PMax}, \
                        cmap=PlotColor)
        viewerTemp.axes.set_xlabel('Position (Dimensionless)', fontdict=Font)
        viewerTemp.axes.set_ylabel('Momentum (Dimensionless)', fontdict=Font)
        matplotlib.pyplot.savefig(path+targetName)
        matplotlib.pyplot.close(viewerTemp.id)


# Calculates heat transfer during evolution
def calcHeat(initialDist, responseFunc1, responseFunc2, conjRevResponseFunc):
    probElements = scipy.special.xlogy(responseFunc1,\
                                         responseFunc2/conjRevResponseFunc)
    # Remove spurious NaN's
    probElements[updatedResponseFunctions==0] = 0
    # Integrate over final microstates
    marginalElements = numpy.sum(probElements, axis=1)*Dx*Dp
    # Combine with updated initial distribution and integrate over initial
    # microstates
    temp = initialDist*marginalElements
    deltaQ = numpy.sum(temp)*Dx*Dp
    return deltaQ    


# Conjugates a given distribution
def conjugateDist(initialDist):
    conjDist = numpy.zeros(((2*Nx)**2,))
    for k in range(2*Nx):
        conjDist[2*Nx*k:2*Nx*(k+1)] = \
                                initialDist[2*Nx*(2*Nx-k-1):2*Nx*(2*Nx-k)]
    return conjDist


#############################################################################
#############################################################################


# Find current time for timing purposes
t0 = time.time()

# Create sub-directories for storing evolution of distribution
if (not os.path.exists(pathStr + 'ForwardEvo/')):
    os.makedirs(pathStr + 'ForwardEvo/')
if (not os.path.exists(pathStr + 'ReverseEvoUnupdated/')):
    os.makedirs(pathStr + 'ReverseEvoUnupdated/')
if (not os.path.exists(pathStr + 'ReverseEvoUpdated/')):
    os.makedirs(pathStr + 'ReverseEvoUpdated/')

# Initialize phase-space lattice
# Nx, Dx, XMax, Np, Dp, PMax, and Mesh are Global
Nx = n1*n2
Np = Nx
Dx = 1./n1
Dp = Dx
XMax = Dx*Nx
PMax = Dp*Np
comm = fipy.tools.serial
Mesh = fipy.Grid2D(dx=Dx, dy=Dp, nx=2*Nx, ny=2*Np, communicator=comm)\
                + [[-XMax], [-PMax]]
fipy.dump.write(Mesh, filename=pathStr+'mesh.gz')


# Create and save initial distribution
phi = distributionInitial()
phiInitial = fipy.CellVariable(mesh=Mesh, name=r"\LARGE $\rho_{0}(x,p)$")
phiInitial.setValue(phi.value)
del phi
if(saveResults):
    fipy.tools.dump.write(phiInitial, \
                                filename=pathStr+'initialDistribution.gz')


# Plot initial distribution
genPlots(phiInitial, tStart, saveResults, 'initialDistribution.pdf', pathStr)


# Evolve to final distribution
phi = fipy.CellVariable(mesh=Mesh, name=r"\LARGE $\rho(x,p)$")
phi.setValue(phiInitial.value)
phi = evolution(phi, tStart, tFinal, pathStr+'ForwardEvo/', saveResults)
phiFinal = fipy.CellVariable(mesh=Mesh, name=r"\LARGE $\rho_{f}(x,p)$")
phiFinal.setValue(phi.value)
del phi
if(saveResults):
    fipy.tools.dump.write(phiFinal, \
                            filename=pathStr+'finalDistribution.gz')


# Display final distribution
genPlots(phiFinal, tFinal, saveResults, 'finalDistribution.pdf', pathStr)


# Find cycled distribution
finalProb = phiFinal.value
conjProb = conjugateDist(finalProb)
phiConj = fipy.CellVariable(mesh=Mesh)
phiConj.setValue(conjProb)
del finalProb,conjProb
phiConjFinal = evolution(phiConj, tFinal, tStart, \
                                pathStr+'ReverseEvoUnupdated/', saveResults)
del phiConj
conjProb = phiConjFinal.value
cycledProb = conjugateDist(conjProb)
phiCycled = fipy.CellVariable(mesh=Mesh,
                            name=r"\LARGE $\widetilde{\rho}(x,p)$")
phiCycled.setValue(cycledProb)
del conjProb, cycledProb, phiConjFinal


# Display and save cycled distribution
genPlots(phiCycled, tStart, saveResults, 'cycledDistribution.pdf', pathStr)
if(saveResults):
    fipy.tools.dump.write(phiCycled, \
                                filename=pathStr+'cycledDistribution.gz')

# Perform update on final distribution
finalValues = phiFinal.value
updatedValues = updateDistribution(finalValues, mX, mP, sigmaX, sigmaP)
phiFinalUpdated = fipy.CellVariable(mesh=Mesh, 
                                        name=r"\LARGE $\rho_{f|m}(x,p)$")
phiFinalUpdated.setValue(updatedValues)
del finalValues,updatedValues
if(saveResults):
    fipy.tools.dump.write(phiFinalUpdated, \
                            filename=pathStr+'updatedFinalDistribution.gz')


# Display updated final distribution
genPlots(phiFinalUpdated, tFinal, saveResults, \
                                    'updatedFinalDistribution.pdf', pathStr)


# Find updated cycled distribution
finalProb = phiFinalUpdated.value
conjProb = conjugateDist(finalProb)
phiConj = fipy.CellVariable(mesh=Mesh)
phiConj.setValue(conjProb)
del finalProb,conjProb
phiConjFinal = evolution(phiConj, tFinal, tStart,\
                             pathStr+'ReverseEvoUpdated/', saveResults)
del phiConj
conjProb = phiConjFinal.value
cycledProb = conjugateDist(conjProb)
phiUpdatedCycled = fipy.CellVariable(mesh=Mesh,
                            name=r"\LARGE $\widetilde{\rho}_{m}(x,p)$")
phiUpdatedCycled.setValue(cycledProb)
del conjProb, cycledProb, phiConjFinal


# Display and save updated cycled distribution
genPlots(phiUpdatedCycled, tStart, saveResults, \
                                'updatedCycledDistribution.pdf', pathStr)
if(saveResults):
    fipy.tools.dump.write(phiUpdatedCycled, \
                            filename=pathStr+'updatedCycledDistribution.gz')

# Checkpoint 1
t1 = time.time()
print('CheckPoint #1: ' + repr(t1-t0))


# Find all impulse response functions using multiprocessing
cpuCount = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cpuCount)
responseStorePar = pool.map(impulseResponse, \
                            [[tStart, tFinal, k] for k in range((2*Nx)**2)])
pool.close()
pool.join()
del pool
responseStore = numpy.zeros(((2*Nx)**2,(2*Nx)**2))
for k in range((2*Nx)**2):
    responseStore[k,:] = responseStorePar[k].value
del responseStorePar
if(saveResults):
    numpy.save(pathStr+'impulseResponses', responseStore)


# Checkpoint 2
t2 = time.time()
print('CheckPoint #2: ' + repr(t2-t1))


# Update all impulse response functions
updatedResponseFunctions = numpy.zeros(((2*Nx)**2,(2*Nx)**2))
for k in range((2*Nx)**2):
    response = responseStore[k,:]
    updatedResponse = updateDistribution(response, mX, mP, sigmaX, sigmaP)
    updatedResponseFunctions[k,:] = updatedResponse
    del response, updatedResponse
if(saveResults):
    numpy.save(pathStr+'updatedImpulseResponses', updatedResponseFunctions)


# Find updated initial distribution
unupdatedValues = phiInitial.value
updatedInitial = updateInitial(unupdatedValues, responseStore,\
                                            mX, mP, sigmaX, sigmaP)
phiInitialUpdated = fipy.CellVariable(mesh=Mesh,
                                      name=r"\LARGE $\rho_{0|m}(x,p)$")
phiInitialUpdated.setValue(updatedInitial)
if(saveResults):
    fipy.tools.dump.write(phiInitialUpdated, \
                            filename=pathStr+'updatedInitialDistribution.gz')


# Display the updated initial distribution
genPlots(phiInitialUpdated, tStart, saveResults, \
                                'updatedInitialDistribution.pdf', pathStr)


# Calculate reverse protocol impulse response functions
pool = multiprocessing.Pool(processes=cpuCount)
revResponseStorePar = pool.map(impulseResponse, \
                            [[tFinal, tStart, k] for k in range((2*Nx)**2)])
pool.close()
pool.join()
del pool
revResponseStore = numpy.zeros(((2*Nx)**2,(2*Nx)**2))
for k in range((2*Nx)**2):
    revResponseStore[k,:] = revResponseStorePar[k].value
del revResponseStorePar


# Conjugate the reversed impulse functions
# revResponseStore[x,x'] is currently equivalent to Pi_R[x->x'], instead 
# we want revResponseStore[x,x'] = Pi_R[x'_bar->x_bar]
temp = numpy.zeros(((2*Nx)**2,(2*Nx)**2))
conjRevResStore = numpy.zeros(((2*Nx)**2,(2*Nx)**2))
for k in range(2*Nx):
    temp[2*Nx*k:2*Nx*(k+1),:] = \
                        revResponseStore[2*Nx*(2*Nx-k-1):2*Nx*(2*Nx-k),:]
for k in range(2*Nx):
    conjRevResStore[:,2*Nx*k:2*Nx*(k+1)] = \
                        temp[:,2*Nx*(2*Nx-k-1):2*Nx*(2*Nx-k)]
del temp
conjRevResStore = conjRevResStore.transpose()
if(saveResults):
    numpy.save(pathStr+'conjImpulseResponses', conjRevResStore)


# Checkpoint 3
t3 = time.time()
print('CheckPoint #3: ' + repr(t3-t0))


# Calculate Entropies
values = phiInitial.value
sInit = -numpy.sum(scipy.special.xlogy(values,values))*Dx*Dp
values = phiInitialUpdated.value
sInitUpdate = -numpy.sum(scipy.special.xlogy(values,values))*Dx*Dp
values = phiFinal.value
sFinal = -numpy.sum(scipy.special.xlogy(values,values))*Dx*Dp
values = phiFinalUpdated.value
sFinalUpdate = -numpy.sum(scipy.special.xlogy(values,values))*Dx*Dp


# Calculate Relative Entropies
sRelativeInit = relEntropy(phiInitialUpdated, phiInitial)
sRelativeFinal = relEntropy(phiFinalUpdated, phiFinal)


# Calculate Irreversibilities
initProbUpdated = phiInitialUpdated.value
cycProbUpdated = phiUpdatedCycled.value
sIrrUpdated = numpy.sum(scipy.special.xlogy(initProbUpdated, \
                                    initProbUpdated/cycProbUpdated))*Dx*Dp
del initProbUpdated, cycProbUpdated
initProb = phiInitial.value
cycProb = phiCycled.value
sIrrUnupdated = numpy.sum(scipy.special.xlogy(initProb, \
                                                initProb/cycProb))*Dx*Dp
del initProb, cycProb


# Calculate updated and unupdated heat transfers
initialDist = phiInitial.value
deltaQf = calcHeat(initialDist, responseStore, \
                                        responseStore, conjRevResStore)
initialUpdated = phiInitialUpdated.value
deltaQfm = calcHeat(initialUpdated, updatedResponseFunctions, \
                                        responseStore, conjRevResStore)


# Check generalized Jarzynski equalities
pInit = phiInitial.value
pInitUpdate = phiInitialUpdated.value
pFinal = phiFinal.value
pFinalUpdate = phiFinalUpdated.value
pCycled = phiCycled.value
pUpdatedCycled = phiUpdatedCycled.value
# Define joint probability distributions
Pr = conjRevResStore*pFinal.reshape((1,pFinal.size))
Pf = pInit.reshape((pInit.size,1))*responseStore
Prm = conjRevResStore*pFinalUpdate.reshape((1,pFinalUpdate.size))
Pfm = pInitUpdate.reshape((pInitUpdate.size,1))*updatedResponseFunctions
# <Pr/Pf>f = a
temp = Pr
temp[Pf==0] = 0
a = numpy.sum(numpy.sum(temp)*Dx*Dp)*Dx*Dp
# <Pr|m/Pf|m>f|m = b
temp = Prm
temp[Pfm==0] = 0
b = numpy.sum(numpy.sum(temp)*Dx*Dp)*Dx*Dp
# <Pr*p0/Pf*p_tilde>f = c
temp = Pr*pInit.reshape((pInit.size,1))/pCycled.reshape((pCycled.size,1))
temp[Pf==0] = 0
c = numpy.sum(numpy.sum(temp)*Dx*Dp)*Dx*Dp
# <Pr|m*p0|m/Pf|m*p_tilde|m>f|m = d
pIUtemp = pInitUpdate.reshape((pInitUpdate.size,1))
pTMtemp = pUpdatedCycled.reshape((pUpdatedCycled.size,1))
temp = Prm*pIUtemp/pTMtemp
temp[Pfm==0] = 0
d = numpy.sum(numpy.sum(temp)*Dx*Dp)*Dx*Dp


# Display all results

# Unupdated Quantities
print('-----Unupdated Quantities-----')
s1 = r'S[p_{0}]: '
print(s1 + '%.10f' %sInit)
s1 = r'S[p_{f}]: '
print(s1 + '%.10f' %sFinal)
s1 = r'\Delta S: '
deltaS = sFinal-sInit
print(s1 + '%.10f' %deltaS)
s1 = r'<Q>_{F}: '
print(s1 + '%.10f' %deltaQf)
s1 = r'\Delta S + <Q>_{F}: '
lhs = deltaS + deltaQf
print(s1 + '%.10f' %lhs)
s1 = r'D(p_{0} | \tilde{p}): '
print(s1 + '%.10f' %sIrrUnupdated)

# Updated Quantities
print('-----Updated Quantities-----')
s1 = r'S[p_{0|m}]: '
print(s1 + '%.10f' %sInitUpdate)
s1 = r'S[p_{f|m}]: '
print(s1 + '%.10f' %sFinalUpdate)
s1 = r'\Delta S_{m}: '
deltaSm = sFinalUpdate - sInit
print(s1 + '%.10f' %deltaSm)
s1 = r'D(p_{0|m} | p_{0}): '
print(s1 + '%.10f' %sRelativeInit)
s1 = r'D(p_{f|m} | p_{f}): '
print(s1 + '%.10f' %sRelativeFinal)
s1 = r'H(p_{0|m}, p_{0}): '
H0 = sInitUpdate + sRelativeInit
print(s1 + '%.10f' %H0)
s1 = r'H(p_{f|m}, p_{f}): '
Hf = sFinalUpdate + sRelativeFinal
print(s1 + '%.10f' %Hf)
s1 = r'\Delta H: '
deltaH = Hf - H0
print(s1 + '%.10f' %deltaH)
s1 = r'<Q>_{F|m}: '
print(s1 + '%.10f' %deltaQfm)
s1 = r'\Delta H + <Q>_{F|m}: '
lhs = deltaH + deltaQfm
print(s1 + '%.10f' %lhs)
s1 = r'D(p_{0|m} | \tilde{p}_{m}): '
print(s1 + '%.10f' %sIrrUpdated)
s1 = r'LHS of BSL: '
LHS = deltaSm + deltaQfm
print(s1 + '%.10f' %LHS)
s1 = r'RHS of BSL: '
RHS = sIrrUpdated - sRelativeFinal - sInit + H0
print(s1 + '%.10f' %RHS)
s1 = r'|(LHS-RHS)/LHS|: '
frac = numpy.abs((LHS-RHS)/LHS)
print(s1 + '%.10f' %frac)

# Check generalized Jarzynski equalities
print('Generalized Jarzynski Equalities:')
print(r'<P_{r}/P_{f}>_{f} = a = ' + '%.10f' %a)
print(r'<P_{r|m}/P_{f|m}>_{f|m} = b = ' + '%.10f' %b)
print(r'<p_{0}*P_{r}/P_{f}*\tilde{p}>_{f} = c = ' + '%.10f' %c)
print(r'<p_{0|m}*P_{r|m}/P_{f|m}*\tilde{p}_{m}>_{f|m} = d = ' + '%10f' %d)
