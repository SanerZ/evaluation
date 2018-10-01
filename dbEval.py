# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:37:21 2018

@author: ADMIN
"""
import sys
sys.path.append('D:/work/workspace')

from argparse import ArgumentParser
import matplotlib.pyplot as plt

import numpy as np
from pathlib2 import Path
from collections import namedtuple

from eval_exps import main

def load_cfg(algFile):
    with open(algFile,'r') as f:
        def extractName(sourceFile):
            sourceFile = sourceFile.strip()
            targetFile = Path(sourceFile).stem
            # targetFile, _ = osp.splitext(osp.basename(sourceFile))
            return targetFile
        
        algNms = [extractName(l) for l in f.readlines()]
        
    return algNms

class Config:
    def __init__(self, projDir, algFile, dataDir):

        """
        Global parameters
        """
        self.resDir = Path(projDir)
        self.labels = [
                'face',
                ]

       
        """
        List of experiment settings: {name: [hr, vr, ar, overlap, filter]}
        %  name     - experiment name
        %  hr       - height range to test
        %  vr       - visibility range to test
        %  ar       - aspect ratio range to test
        %  overlap  - overlap threshold for evaluation
        %  filter   - expanded filtering (see 3.3 in PAMI11)
        """
        inf = np.inf
        expParams = namedtuple('ExpParameters', ['hr', 'vr', 'ar', 'overlap', 'filter', 'visible'])
        
        self.expsDict = {
            'Reasonable':   expParams([0.06,0.15],  [.01,inf],   0,  .45, 1., 1),
            'All'       :   expParams([0,inf],      [.01,inf],   0,  .45, 1., 1),
            # 'small'     :   expParams([0,0.07],     [.01,inf],   0,  .45, 1., 1),
            # 'medium'    :   expParams([0.07,0.1],   [.01,inf],   0,  .45, 1., 1),
        }
        
        
        self.algNames = load_cfg(algFile)
        
        self.ds_cfgDir = dataDir
        dsParams = namedtuple('DataSetParameters', ['resize', 'visible'])
        self.dsDict = {
                'benchmark_1w_gt' :  dsParams(0, 0),
                'anfang_bg1w_gt'  :  dsParams(0, 0),
                'wider'           :  dsParams(1, 1),
                'wider_fddb'      :  dsParams(1, 1), 
        }
        
        self.bgName = 'lst'
        
        self.thr = 0.3
        self.wrongDir = self.resDir/'det_wrong'
        self.evShow = 1
        
        """
        Remaining parameters and constants
        """
        self.visible = 0
        self.reapply = [0, 0, 0]         # if true create all the .npy file from scratch
        self.aspectRatio = 1.            # default aspect ratio for all bbs
        self.bnds = [5, 5, 635, 475]     # discard bbs outside this pixel range
        self.resize = 100./128           # rescale height of each box by resize
        self.plotRoc = 0                 # if true plot ROC else PR curves
        self.plotAlg = 0                 # if true one plot per alg else one plot per exp
        self.plotNum = 15                # only show best plotNum curves (and VJ and HOG)
        #samples = 10.^(-2:.25:0);  # samples for computing area under the curve
        #lims = [2e-4 50 .035 1];   # axis limits for ROC plots
        self.bbsShow = 1                 # if true displays sample bbs for each alg/exp
        self.bbsType = 'fp'              # type of bbs to display (fp/tp/fn/dt)



if __name__=='__main__':
    projDir = 'D:/work/workspace/proj/evaluation'
    dataDir = 'D:/work/data'
    
    parser = ArgumentParser()
    parser.add_argument('--algFile', '-a', default='algNames.lst')
    parser.add_argument('--server', '-s', action='store_true', help='server environment -- no display')
    parser.add_argument('--reapply', '-r', action='store_true', help='reload dt- and ev-')
    parser.add_argument('--gtReapply', '-g', action='store_true', help='reload gt- npy file')
    parser.add_argument('--visible', '-v', action='store_true', help='output pictures with bounding boxes')
    
    args = parser.parse_args()  
    
    if args.server:
        plt.switch_backend('agg')
        
    cfg = Config(projDir, args.algFile, dataDir)
    if args.reapply:
        cfg.reapply |= np.array((1, 0, 1))
    if args.gtReapply:
        cfg.reapply |= np.array((0, 1, 0))
        
    if args.visible:
        cfg.visible = 1
        
    main(cfg)
