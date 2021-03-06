# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:09:50 2018

@author: ADMIN
"""

"""
convert from matlab file

caltech_tool_kit/dbEval.m

"""


import numpy as np
import os.path as osp
from pathlib2 import Path
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt

from detectron.ds_factory import ds_factory, ds_config

from detectron.bbs_eval import evalRes, compRoc, ref_threshold, output_bounding_boxes
from detectron.bbs_utils import boxResize

from detectron.loggers import lcfg, pl, ps

def compRef(cfg):
    bgName = cfg.resDir/cfg.bgName
    dts = loadDts(cfg, bgName.as_posix())
    ref = dict()
    if len(dts) == 0:
        return ref
    
    lcfg({'logfile': (bgName/'Ref').as_posix()})
    print('\nExp reference threshold:')
    print('\n{0:<28}\t{1[0]:<8g}\t{1[1]:<8g}\t{1[2]:<8g}\t{1[3]:<8g}'.\
                  format('', 0.1**np.arange(4,0,-1)))
    for algNm in cfg.algNames:
        det = [np.column_stack((d, [i]*len(d))) for i,d in enumerate(dts[algNm])]
        nImg = len(det)
        det = np.concatenate(det, 0)
        det = det[np.any(det[:,:4], axis=1)]
        det = det[np.argsort(-det[:,4])]
        ids, score = det[:,-1], det[:, 4]
        fp = np.ones(len(ids)).astype(bool)
        ref_thr, ref_idx = ref_threshold(ids, score, fp, nImg)
        ref[algNm] = ref_thr
        pl.info('{0:^28}\t{1[0]:<8.3}\t{1[1]:<8.3}\t{1[2]:<8.3}\t{1[3]:<8.3}'.
               format(algNm, ref_thr))
    print('\n\n')
    return ref

def loadDts(cfg, pltName):
    print('\nLoading detections: %s' % pltName)
    dts = defaultdict(list)
    for d, a in enumerate(cfg.algNames):
        aNm = osp.join(pltName, 'dt-'+a)
        # already resize the bbox
        if not cfg.reapply[0] and osp.exists(aNm+'.npy'):
            dts[a].extend(np.load(aNm+'.npy'))
            continue
        
        if not osp.exists(aNm+'.txt'):
            continue
        # load from txt file format as LTWH
        print('\tAlgorithm #%d: %s' % (d, a))
        dt0 = np.loadtxt(aNm+'.txt').reshape((-1,6))
        ids = dt0[:,0].astype(int)
        dt = [dt0[ids==i,1:] for i in range(max(ids)+1)]
        dt = boxResize(dt, 1, 0, cfg.aspectRatio)
        dts[a].extend(dt)
        np.save(aNm+'.npy', dt)
    
    return dts#, dts.keys()

def filterGtFun(bb, bbv, hr, vr, ar, bnds, aspectRatio):
#    p = lbl in labels                           # lbl of the bounding box
    h = bb[3]                   
    p = h>=hr[0] and h<hr[1]              # height range
    vf = (bbv[2]*bbv[3])/(bb[2]*bb[3]) if np.any(bbv!=0) else np.inf
    p = p and vf>=vr[0] and vf<=vr[1]           # visibility range
    if ar!=0:                                   # aspect ratio range
        p = p and np.sign(ar)*abs(bb[2]/bb[3]-aspectRatio)<ar
    # bns ???
    return p


# get gt_roidb according to dtName   
# return gts format as LTWH       
def loadGts(cfg, pltName):
    print('\nLoading ground truth: %s' % pltName)
    Path(pltName).mkdir(parents=True, exist_ok=True)
    
    gts = defaultdict(list)
    dtNm = osp.basename(pltName)
    gt0 = ds_factory(dtNm)

    rz = cfg.resize if cfg.dsDict[dtNm].resize else 1
    for g, e in enumerate(cfg.expsDict):
        gNm = osp.join(pltName, 'gt-'+e) + '.npy'
        if not cfg.reapply[1] and osp.exists(gNm):
            gts[e].extend(np.load(gNm))
            continue
        
        print('\tExperiment #%d: %s' % (g, e))
        exp = cfg.expsDict[e]
        filterGt = partial(filterGtFun, hr=exp.hr, vr=exp.vr, ar=exp.ar, \
                           bnds=cfg.bnds, aspectRatio=cfg.aspectRatio)

        gt = gt0.gt_filter(labels=cfg.labels, filterGt=filterGt)
        gt = boxResize(gt, rz, 0, cfg.aspectRatio)
        gts[e].extend(gt)
        np.save(gNm, gt)
      
    gNm = osp.join(pltName, 'gt-sides.npy')
    if not cfg.reapply[1] and osp.exists(gNm):
        gt_sides = np.load(gNm)
    else:
        gt_sides = np.maximum(gt0.heights,gt0.widths)
        np.save(gNm, gt_sides)
        
    return gts, gt_sides

def evalAlgs(cfg, pltName, gts, dts, gt_sides):
    print('\nEvaluating: %s' % pltName)
    res = defaultdict(list)
    for g, expNm in enumerate(cfg.expsDict):
        for d, algNm in enumerate(cfg.algNames):
            # check whether exsits ev-Reasonable-FCN.npy
            evNm = osp.join(pltName, ''.join(['ev-',expNm,'-',algNm,'.npy']))
            if not cfg.reapply[2] and osp.exists(evNm):
                r = np.load(evNm)
                res[expNm, algNm].extend(r)
                continue

            gt, dt = gts[expNm], dts[algNm]
            if len(dt)==0:
                continue
            # evalRes from gt and dt
            print('\tExp %d/%d, Alg %d/%d,: %s/%s' % \
                  (g, len(cfg.expsDict), d, len(cfg.algNames), expNm, algNm))
            # filter the detection results
            exp = cfg.expsDict[expNm]
            hr = np.multiply(exp.hr, [1./exp.filter, exp.filter])
            dt = [bx[np.where(np.all((bx[:,3]>hr[0]*gt_sides[i], \
                                      bx[:,3]<hr[1]*gt_sides[i]), axis=0))] \
                    for i, bx in enumerate(dt)]
            r = evalRes(gt, dt, ovthresh=exp.overlap)
            res[expNm, algNm].extend(r)
            np.save(evNm, r)
    return res    

def plotExps(cfg, res, plotName, ref_score=None):
    """
    % Plot all ROC or PR curves.
    %
    % INPUTS
    %  res      - output of evalAlgs
    %  plotRoc  - if true plot ROC else PR curves
    %  plotAlg  - if true one plot per alg else one plot per exp
    %  plotNum  - only show best plotNum curves 
    %  plotName - filename for saving plots
    """
    # Compute (xs, ys) for every exp/alg
    print('\nPlotting: %s' % plotName)
    roc = defaultdict(list)
    ref = 0.1**np.arange(4,0,-1) 
    for (e, a), (g, d) in res.items():
        g, d = list(g), list(d)
        try:
            ref_thr = ref_score[a]
        except:
            ref_thr = None
        
        print('\tExp %d/%d, Alg %d/%d,: %s/%s' % \
             (cfg.expsDict.keys().index(e), len(cfg.expsDict), \
              cfg.algNames.index(a), len(cfg.algNames), e, a))
        
        r = list(compRoc(g, d, ref_score=ref_thr))
        if cfg.plotAlg:
            roc[a].append([e]+r)    # [[expNm, rec, prec, ap, recpi, ref_thr],[]...]
        else:
            roc[e].append([a]+r)    # [[algNm, rec, prec, ap, recpi, ref_thr],[]...]
    
        
    print('\n')
    # Generate plots
    for k, v in roc.items():
        colors = [np.maximum(np.mod(np.array([78,121,42])*(i+1),255)/255.,.3) \
                  for i in range(len(v))]
        if cfg.plotAlg:
            # TODO: 
            pass
        else:
            v.sort(key=lambda x:x[3], reverse=True)  # 为了显示的legend分数从高到低
        # plot curves and finalize display
        plt.figure(figsize=[10,8])
        saveName = plotName + '_' + k
        
        lcfg({'logfile': saveName})
        ps.info('\n\n\t')
        pl.info('{0:^28}\t{1[0]:<11g}\t{1[1]:<11g}\t{1[2]:<11g}\t{1[3]:<11g}\t{2:^}'.
                       format('reference', ref, 'mAP'))

            
        for i, p in enumerate(v):
            plt.plot(p[1], p[2], color=colors[i], ls=(i%2+1)*'-', \
                     label='%.2f%% %s' % (p[3]*100, p[0]))
            pl.info('{0:^28}\t{1[0]:<11.3%}\t{1[1]:<11.3%}\t{1[2]:<11.3%}\t{1[3]:<11.3%}\t{2:<.3%}'.
                       format(p[0], p[4], p[3]))
            if ref_thr is None:
                pl.info('{0:^28}\t{1[0]:<8.3}\t{1[1]:<8.3}\t{1[2]:<8.3}\t{1[3]:<8.3}'.
                       format('', p[5]))
            
        plt.legend()
        
        if cfg.plotRoc:
            # TODO:
            pass
        else:
#            plt.xlim(0,1)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        plt.title(osp.basename(saveName))
        
        plt.savefig(saveName)
        plt.show()

def drawBoxes(cfg, dtNm, res):
    # Draw box for each picture and save
    print('\nDisplaying evaluation results for dataset: %s' % dtNm)
    show_params = {'thr'      : cfg.thr,
                   'evShow'   : cfg.evShow,
                   'outpath'  : None,}
    gt = ds_factory(dtNm)
    for (e, a), (g, d) in res.items():
        if not cfg.expsDict[e].visible:
            continue
        
        g, d = list(g), list(d)
        nImg = len(g)
        assert len(d)==nImg
               
        for idx in range(nImg):
            raw_img = plt.imread(gt.image_path_at(idx))
            imgpath = cfg.wrongDir/dtNm/a/e/('%05d.jpg' % idx)
            imgpath.parent.mkdir(parents=True, exist_ok=True)
            show_params['outpath'] = imgpath.as_posix()
            output_bounding_boxes(raw_img, g[idx], d[idx], **show_params)

def main(cfg):
    ds_config.read((Path(cfg.ds_cfgDir)/'datasets.conf').as_posix())
    
    ref_score = compRef(cfg)    
    for dtNm in cfg.dsDict.keys(): 
        pltName = (cfg.resDir/dtNm).as_posix()
        plotName = (cfg.resDir/'results'/dtNm).as_posix()
        # load detections and ground truth and evaluate
        dts = loadDts(cfg, pltName) #, algNms 
        gts, gt_sides = loadGts(cfg, pltName)
        res = evalAlgs(cfg, pltName, gts, dts, gt_sides)
        # plot curves and bbs
        plotExps(cfg, res, plotName, ref_score)
        if cfg.visible and cfg.dsDict[dtNm].visible:
            drawBoxes(cfg, dtNm, res)
        

        
        
        