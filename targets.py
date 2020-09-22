import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

rng = np.random.default_rng() #initialize random generator to choose the sequences

def moverule(traj,vertex): 
    
    #this is the subroutine that chooses next moves. Probabilities are hard-coded but could
    #be exposed if necessary.
    
    llen = 0.5*vertex*(3**0.5) #"long" length of triangle (bisecting one leg)
    slen = 0.5*vertex #short length of triangle (probably not used)
    
    #These are the possible moves
    up = np.array((0,vertex))
    upleft = np.array((-llen,slen))
    upright = np.array((llen,slen))
    dn = np.array((0,-vertex))
    dnleft = np.array((-llen,-slen))
    dnright = np.array((llen,-slen))

    prev = traj[-1]-traj[-2]

    if np.all(np.round(prev) == np.round(dn)): #moved straight down
        #print('dn')
        next_move = rng.choice([dn,dnleft,dnright], p=[0.2, 0.4, 0.4])

    elif np.all(np.round(prev) == np.round(dnleft)): #moved down and to the left
        #print('dnleft')
        next_move = rng.choice([dn,dnleft,upleft], p=[0.4, 0.2, 0.4])

    elif np.all(np.round(prev) == np.round(dnright)): #moved down and to the right
        #print('dnright')
        next_move = rng.choice([dn,dnright,upright], p=[0.4, 0.4, 0.2])

    elif np.all(np.round(prev) == np.round(up)): #moved up
        #print('up')
        next_move = rng.choice([up,upleft,upright], p=[0.2, 0.4, 0.4])

    elif np.all(np.round(prev) == np.round(upleft)): #moved up and to the left
        #print('upleft')
        next_move = rng.choice([up,upleft,dnleft], p=[0.4, 0.2, 0.4])

    elif np.all(np.round(prev) == np.round(upright)): #moved up and to the right
        #print('upright')
        next_move = rng.choice([up,upright,dnright], p=[0.4, 0.2, 0.4])

    return next_move

def gen_cands(n,seq_length,vertex):
    
    #this subroutine generates n candidate sequences of length seq_length. All moves are of 
    #"vertex" length.
    
    llen = 0.5*vertex*(3**0.5) #"long" length of triangle (bisecting one leg)
    slen = 0.5*vertex #short length of triangle (probably not used)
    
    #catalog of possible moves from any node, this is distance to move to get to the 
    #next node, preserving move length of "vertex"

    up = np.array((0,vertex))
    upleft = np.array((-llen,slen))
    upright = np.array((llen,slen))
    dn = np.array((0,-vertex))
    dnleft = np.array((-llen,-slen))
    dnright = np.array((llen,-slen))
    
    cand = np.zeros((seq_length+1,2)) #sequence length +1 because of first target
    
    for k in np.arange(n): #generate n candidate sequences
        
        #Intitalize with a random move from 0,0 (this coule be relaxed)
        traj = np.vstack((np.zeros(2),rng.choice([up,dn,upright,upleft,dn,dnleft,dnright])))
        
        for i in np.arange(1,seq_length):
            
            #build the rest of the trajectory
            traj = np.vstack((traj,traj[-1]+moverule(traj,vertex)))
            
        cand = np.dstack((cand,traj)) #add each candidate to the 3D array
    
    return cand[:,:,1:n]

def gen_grid(x_range,y_range,vertex):
    
    #generate the target space with dimensions approximately tgrid_x,tgrid_y
    #one moderately tricky thing is we want 0,0 to be in the grid
    
    xs = np.arange(start=-x_range-.5*3**.5,stop=x_range,step=vertex*3**.5) #beginning x array
    if len(xs) % 2 == 0: #check number of columns
        xs = xs[1:len(xs)] #coerce to an odd number of columns
    xs = xs-np.median(xs) # we want origin at actual 0,0
    
    ys = np.arange(start=-y_range,stop=y_range,step=vertex)
    if len(ys) % 2 == 0: #check number of rows (half the number actually)
        ys = ys[1:len(ys)] #coerce to an odd number of columns
    ys = ys-np.median(ys) # we want origin at actual 0,0
    
    targs = np.zeros((2,len(xs))) #initialize the targets array
    
    for i in ys:
    
        t_int1 = np.vstack((xs,np.full((1,len(xs)),i))) #a single row of "major axis" targets

        t_int2 = np.vstack((xs+(vertex/2)*3**.5,np.full((1,len(xs)),i+vertex/2))) #a single row of "minor axis" targets

        targs = np.hstack((targs,t_int1))
        targs = np.hstack((targs,t_int2))
        
    return np.round(targs[:,len(xs):np.shape(targs)[1]],3)

def compactness(cand,xmin,xmax,ymin,ymax):
    
    #choose sequences which satisfy some criterion about how long they are (to fit within
    #robot workspace)
    print('checking compactness with x range: ' + str(xmin) + ', ' + str(xmax))
    print('checking compactness with y range: ' + str(ymin) + ', ' + str(ymax))
    compact = []
    for i in np.arange(cand.shape[2]):
            
        if (np.max(cand[:,0,i]) < xmax):
            if (np.min(cand[:,0,i]) > xmin):
                if (np.max(cand[:,1,i]) < ymax):
                    if (np.min(cand[:,1,i]) > ymin):
                
                        compact += [i]

    
    return compact

def targorder(target_positions,xy_sequences,adjust_index):
    
    #Once a sequence has been generated in xy, we want to match it to targets in the target table to get a
    #sequence of targets defined by their presence in the target table.
    #adjust_index because matlab/KINARM is 1-indexed. Set to 0 if you don't need it.
    
    #This version computers euclidian distance between location and the target to find
    #the referenced target.
    
    tseq = []
    
    for i in xy_sequences:
        
        difs = []
        
        for j in np.arange(len(target_positions[1])):
            
            temp_dif = [i - grid[:,j]]
            difs += [(temp_dif[0][0]**2 + temp_dif[0][1]**2)**0.5]
        
        tseq += [np.argmin(difs) + adjust_index]
                
    return tseq

x_range = int(input('Input X Range '))
y_range = int(input('Input Y Range '))
vertex = int(input('Vertex Length '))
n_candidates = int(input('Number of Candidates to Try '))
seq_length = int(input('Length of Sequence (number of reaches) '))
show = input('Show generated sequences? (y/n) -- warning: function not yet working ')

grid = gen_grid(x_range,y_range,vertex)
print('generated grid')
print('generating candidate sequences...')
cands = gen_cands(n_candidates, seq_length, vertex)
print('generated ' + str(n_candidates) + ' candidate sequences of ' + str(seq_length) + ' length')
compact_seqs = compactness(cands,-x_range/2,x_range/2,-y_range/2,y_range/2)
print('found ' + str(len(compact_seqs)) + ' suitable candidate sequences')


if len(compact_seqs) > 0:
    seqs = []
    for i in compact_seqs:
        seqs += [targorder(grid,cands[:,:,i],1)] #add the index 1 for MATLAB
    np.savetxt('sequences.txt',seqs,fmt='%1.0f', delimiter='\t')
    print('Wrote sequences.')
    np.savetxt('targets.txt',grid.transpose(),fmt='%1.2f', delimiter ='\t')
    print('Wrote target locations.')

    if show == 'y':
        for j in np.arange(len(seqs)):
            plt.plot(grid[0,:][seqs[j]],grid[1,:][seqs[j]])
        plt.scatter(grid[0],grid[1],color = 'k')
        plt.xlim([-x_range,x_range])
        plt.ylim([-y_range,y_range])
        plt.show()

print('Program Ended')