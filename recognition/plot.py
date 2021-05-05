#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import seaborn as sns
import recognition.util as util
import pandas as pd
import os
import csv
import matplotlib.font_manager


#=========================================================================
#
#  This file contains different functions for plotting data           
#
#=========================================================================

def plot_confusion_matrix(cm, classes, run, amount, normalize=False, title=None, cmap='RdBu'):
    """
    Plot class confusion matrix
    """
    np.set_printoptions(precision = 2)
    if not title:
        if normalize:   title = 'Token-based Normalized Confusion Matrix'
        else:   title = 'Confusion matrix, without normalization'
    
    if normalize:   cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print(cm)
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
		yticks=np.arange(cm.shape[0]),
		xticklabels=classes, yticklabels=classes,
		title=title,
		ylabel='True label',
		xlabel='Predicted label')
        
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black")
            
    ax.margins(0.05)
    ax.grid(False)
    # Default margin is 0.05, value 0 means fit
    plt.savefig("Figure/cm/cm_"+run+'_'+amount+".png")

def plot_distribution(l, name, color):
    """
    Plot dataset length distribution

    Args:
        l (list): List of data
        name (list): List of datasets name
        color (list): List of colors
    """    
    sns.set()
    for ll, nn, cc in zip(l,name, color):
        sns.distplot(ll, hist=False, norm_hist=True, label=nn, color=cc)
    plt.xlabel("Sentence length",fontweight='bold', fontsize=15)
    plt.ylabel("Density",fontweight='bold', fontsize=15)
    plt.tight_layout(pad=1.0)
    #plt.tick_params(axis='both', labelsize=12)
    plt.legend( loc='upper right', prop={'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('Figure/distribution.png')
    plt.clf()

def plot_unit():
    """Plot Unit class distribution"""    
    sns.set()
    tempeval = {'DAY': 6885, 'WEEK': 615, 'MONTH': 1860, 'YEAR': 3196, 'QUARTER': 264, 'SEASON': 52, 'HOUR': 579, 'MINUTE': 56, 'SECOND': 4,'REF': 646, 'UNK': 162 }
    pate = {'DAY': 356, 'WEEK': 22, 'MONTH': 11, 'YEAR':0, 'QUARTER':0, 'SEASON':0, 'HOUR': 269, 'MINUTE': 2,  'SECOND':4, 'REF':0, 'UNK': 13  }
    snips = { 'DAY': 446, 'WEEK': 83, 'MONTH': 63, 'YEAR': 64,'QUARTER':0, 'SEASON': 37, 'HOUR': 479,'MINUTE': 59, 'SECOND': 64, 'REF': 50,  'UNK': 16 }
    domain = [tempeval, snips, pate]
    units = [k for k, v in tempeval.items()]
    barWidth = 0.05
    r_list = []
    print([util.get_percentage(x)[0] for x in domain])
    for i in range(11):
        if i==0:    r_list.append(np.arange(len(domain)))
        else:   r_list.append( [x + barWidth for x in r_list[-1]])
        plt.bar(r_list[i], [util.get_percentage(x)[i] for x in domain], width=barWidth, label=units[i])

    plt.xlabel('Datasets', fontweight='bold')
    plt.ylabel('Units [%]', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(domain))], ['TE-3', 'Snips', 'PÂTÉ'])
    ax = plt.gca()
    plt.legend(loc='upper center', ncol=2)
    plt.tight_layout(pad=0.1)
    plt.savefig("Figure/unit_dist.png")
    plt.clf()

def plot_label_dist():
    """Plot TIMEX3/type distribution"""

    sns.set()
    tempeval = {'DATE':12638, 'TIME':222, 'DURATION':1603, 'SET':98} #Tempeval3
    pate = {'DATE':391, 'TIME':265, 'DURATION':96, 'SET':20}
    snips = {'DATE': 422, 'TIME': 259, 'DURATION': 168, 'SET':0}
    domain = [tempeval, snips, pate]

    barWidth = 0.10
    r1 = np.arange(len(domain))
    r3 = [x + barWidth for x in r1]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]

    plt.bar(r1, [util.get_percentage(x)[0] for x in domain], width=barWidth, label='DATE', color='red')
    plt.bar(r3, [util.get_percentage(x)[1] for x in domain], width=barWidth, label = 'TIME', color='green')
    plt.bar(r4, [util.get_percentage(x)[2] for x in domain], width=barWidth, label = 'DURATION', color='blue')
    plt.bar(r5, [util.get_percentage(x)[3] for x in domain], width=barWidth, label = 'SET', color='black')

    plt.xlabel('Datasets', fontweight='bold', fontsize=15)
    plt.ylabel('Types [%]', fontweight='bold', fontsize=15)
    plt.tick_params(axis='both', labelsize=12)
    plt.xticks([r + barWidth for r in range(len(domain))], ['TE-3', 'Snips', 'PÂTÉ'], fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right', prop={'size': 15})
    plt.tight_layout(pad=1.0)
    plt.savefig("Figure/label.png")
    plt.clf()

def train_test_dist():
    """Plot Train/test split"""
    df = pd.DataFrame({'DATE': {'Train': 307/626,
      'Test': 74/145},
     'TIME': {'Train': 223/626,
      'Test': 47/145},
     'DURATION': {'Train': 79/626,
      'Test': 21/145},
     'SET': {'Train': 17/626,
      'Test': 3/145}})

    sns.barplot(x='RF', y='value', hue='index',
                data=df.reset_index().melt(id_vars='index', var_name='RF'))

    plt.xlabel('Classes', fontweight='bold')
    plt.ylabel('Distribution [%]', fontweight='bold')
    plt.legend()

    plt.savefig("Figure/train_test_split.png")
    plt.clf()

def plot_timex():
    """Plot TIMEX length"""
    sns.set()
    tempeval = {0:0, 1: 7714, 2: 4699, 3: 1291, 5: 216, 4: 562, 7: 14, 6: 58, 8: 4, 10: 3}
    tempeval = dict(sorted(tempeval.items()))
    tempeval = util.cumulative_calculate(tempeval)

    PATE = {0: 213, 1: 302, 2: 442, 3: 17, 4: 7, 5:4}
    PATE = dict(sorted(PATE.items()))
    PATE = util.cumulative_calculate(PATE)

    snips = {0: 79, 1: 315, 2: 379, 3: 103, 4: 33, 5: 15 , 6: 1}
    snips = dict(sorted(snips.items()))
    snips = util.cumulative_calculate(snips)

    a, = plt.plot(list(tempeval.keys()), list(tempeval.values()), marker='o', label='TempEval-3', color='tomato')
    b, = plt.plot(list(PATE.keys()), list(PATE.values()), marker='x', label = 'PATE', color='forestgreen')
    e, = plt.plot(list(snips.keys()), list(snips.values()), marker='^', label = 'Snips', color='yellow')

    plt.legend(handles=[a,b,e])
    plt.xlabel("TIMEX length")
    plt.ylabel("Cumulative Distribution")
    plt.savefig("Figure/timex_.png")
    plt.clf()

def plot_position_dist(l, name, color):
    """Plot TIMEX3 position"""
    sns.set()
    for ll, nn, cc in zip(l, name, color):
        plt.plot(range(len(ll)),ll, marker='o', label=nn, color=cc)
    plt.xlabel("Sentence Distribution")
    plt.ylabel("TIMEX position")
    plt.savefig("Figure/timex_pos.png")
    plt.clf()

def show_values(pc, fmt="%.2f", **kw):
    '''
    Util function for plot classification report
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857  By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Util function for plot classification report
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Plot heatmap for classification report
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    #plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def plot_classification_report(classification_report, run, amount, title='Classification report ', cmap='RdBu'):
    """
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    """
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    #print(lines)
    for line in lines[2 : (len(lines) - 1)]:
    	if not line:    continue
    	t = line.strip().split()
    	if len(t) < 2: continue
    	classes.append(t[0])
    	if t[0].startswith("micro") or t[0].startswith("macro"):
    		v = [float(x) for x in t[2: len(t) - 1]]
    	else:
    		v = [float(x) for x in t[1: len(t) - 1]]
    	support.append(int(t[-1]))
    	class_names.append(t[0])
    	plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))
    #plotMat[0], plotMat[1] = plotMat[1], plotMat[0]
    #support[0], support[1] = support[1], support[0]
    #class_names[0], class_names[1] = class_names[1], class_names[0]

    xlabel = 'Metrics'
    ylabel = 'Classes and Averages'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = True
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
    plt.savefig('Figure/class_report/test_plot_classif_report_'+run+'_'+amount+'.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

def plot_results():
    """Plot final evaluation result: PATE"""
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1,5, sharex=True, sharey=True) #2,2
    loc = ['upper left', 'upper right']
    names = []
    heideltime = [59.48, 78.45, 74.15, 0,38.79]
    uwtime = [68.58, 77.06, 75.32, 0, 38.96]
    te3 = [54.29,72.4,56.11,57.01,48.87]
    color_list = ['blue', 'red', 'magenta', 'purple', 'green']
    for i in range(5): #4
        plot_dict = {}
        for file in sorted(os.listdir('output/')):
            if file.startswith('FT'):
                output_list = []
                try:
                    with open('output/'+file+"/results_averaged.csv") as csv_f:
                        csv_reader = csv.reader(csv_f, delimiter=',')
                        header = next(csv_reader)
                        for row in csv_reader:
                            output_list.append(float(row[i]))
                    name = file.replace("FT_","").replace("FT3_","").replace("FT2_","")
                    names.append(name)
                    plot_dict[name] = output_list
                    index = [j+1 for j in range(10)] * 1 #3
                    index.sort()
                    plot_dict['index'] = index
                    plot_dict['TE-3 Simplified'] = [te3[i]]*10
                    plot_dict['HeidelTime'] = [heideltime[i]]*10
                    plot_dict['UW-Time'] = [uwtime[i]]*10
                except:
                    pass
        df = pd.DataFrame.from_dict(plot_dict)
        df['index'] = df['index'] * 10
        sns.set()
        ax = sns.lineplot(x = 'index', y='value',  hue='variable', data=pd.melt(df, ['index']), palette=color_list, ax=axs[i] ) #axs[sub[i]]

        if i==4:
            #legend = ax.legend()
            handles, labels = ax.get_legend_handles_labels()
            ll = []
            for l in labels:
                l = l.replace("Chain","").replace("_"," ").replace(" ", "").replace("Simplified", "")
                if l == 'Snips':    l = 'Snips → DA-Time$_2$ (TE3+Snips)'
                if l == 'Pate':    l = 'PATE → DA-Time$_2$ (TE3+PATE)'
                if l == 'TE-3':    l = 'TE-3 → DA-Time$_2$ (TE3)'
                ll.append(l)
            ax.legend(loc=loc[1], prop={'size': 12}, handles=handles[1:], labels=ll[1:])
        else:
            ax.get_legend().remove()
        ax.grid(True)
        ax.set(xlim=(10,100))
        ax.set(ylim=(35,100))
        ax.set_xticks( [a for a in range(10,110,10)])
        ax.set_xticklabels([a for a in range(10,110,10)], rotation=45)
        ax.yaxis.set_tick_params(which='both', labelbottom=True)
        l = '$'+header[i].split(" ")[1]+'_{'+header[i].split(" ")[0]+'}$'
        ax.set_ylabel("F-score",fontweight='bold', fontsize=15)
        ax.set_xlabel("Fine-tuning data in (%)", fontweight='bold', fontsize=15)
        ax.set_title(l, fontdict = {'fontsize': 15, 'fontweight' : 'bold'})
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*4, Size[1]*1.2, forward=True)
    plt.savefig('Figure/output_chain.png')

