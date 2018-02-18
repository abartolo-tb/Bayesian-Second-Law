# -*- coding: utf-8 -*-
"""
DocString
"""

# Import modules
import matplotlib
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 18})
matplotlib.use('Agg')
import matplotlib.cm
import os
import os.path
import fipy

######################### User Specified Parameters #########################

# Directory path to where data is saved
pathStr = '/home/tony/Desktop/Final-Versions/Drag-1/'

# Set font used in plots (Global)
Font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

# Color scheme to be used in plots. (Global)
PlotColor = matplotlib.cm.binary


#############################################################################
######################### DO NOT EDIT BELOW HERE ############################
#############################################################################

############################### Functions ###################################

def genPlots(phi, targetName, path):
    viewerTemp = fipy.Matplotlib2DViewer(vars=phi, title='',\
                            limits={'xmin':-XMax, 'xmax':XMax, 
                                    'ymin':-PMax, 'ymax':PMax}, cmap=PlotColor,
                                    datamin=0, datamax=0.2)
    viewerTemp.axes.set_xlabel('Position (Dimensionless)', fontdict=Font)
    viewerTemp.axes.set_ylabel('Momentum (Dimensionless)', fontdict=Font)
    viewerTemp.setLimits()
    matplotlib.pyplot.savefig(path+targetName)
    matplotlib.pyplot.close(viewerTemp.id)

#############################################################################
#############################################################################

if (not os.path.exists(pathStr + 'RePlotted/')):
    os.makedirs(pathStr + 'RePlotted/')

# Load Mesh
Mesh = fipy.tools.dump.read(pathStr+'mesh.gz')

# Load Distributions
phiInitial = fipy.tools.dump.read(pathStr+'initialDistribution.gz')
phiInitialUpdated = fipy.tools.dump.read(pathStr+\
                                            'updatedInitialDistribution.gz')
phiFinal = fipy.tools.dump.read(pathStr+'finalDistribution.gz')
phiFinalUpdated = fipy.tools.dump.read(pathStr+'updatedFinalDistribution.gz')
phiCycled = fipy.tools.dump.read(pathStr+'cycledDistribution.gz')
phiUpdatedCycled = fipy.tools.dump.read(pathStr+\
                                        'updatedCycledDistribution.gz')

# Rename as needed
phiFinal.name = r"\LARGE $\rho_{\tau}(x,p)$"
phiFinalUpdated.name = r"\LARGE $\rho_{\tau|m}(x,p)$"

# Extract Mesh Properties
Nx = Mesh.nx/2
Np = Mesh.ny/2
Dx = Mesh.dx
Dp = Mesh.dy
XMax = Dx*Nx
PMax = Dp*Np

# Generate all plots
pathStr = pathStr + 'RePlotted/'
genPlots(phiInitial, 'initialDistribution.pdf', pathStr)
genPlots(phiFinal, 'finalDistribution.pdf', pathStr)
genPlots(phiInitialUpdated, 'updatedInitialDistribution.pdf', pathStr)
genPlots(phiFinalUpdated, 'updatedFinalDistribution.pdf', pathStr)
genPlots(phiCycled, 'cycledDistribution.pdf', pathStr)
genPlots(phiUpdatedCycled, 'updatedCycledDistribution.pdf', pathStr)
