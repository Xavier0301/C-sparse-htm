#ifndef NETWORK_H
#define NETWORK_H

#include "types.h"
#include "sparse.h"
#include "dense.h"

/* Here is the computation done for inference:

This whole shit is pretty complex
There's stuff going all around, no clean simple abstraction
That's very far from the very high level concept of a graph/network
That's because we keep adding complex shit into it.
I think I need to reduce all this complexity into simple things
But maybe the complexity is inevitable, the only "simple" thing about 

Predicted -> Active:
    1. Activate a sparse set of minicolumns
    2. For all cells in active minicols: 
            if the cell was predicted make it active and winning
            if no cell was predicted make all active and select winning based on rule X

Active -> Predicted:
    1. For all active cells, increment segments it is connected to as an input
        And if the incremented count crosses a threshold, mark cells as predicted

Here is the computation done for learning:

Predicted -> Active: 

The data structures we need are the following:

Predicted -> Active:
    This requires that we know which cells were predicted
        => Store a spvec of predicted cells
    This also requires that we know which cells are connected to which cells
    In particular, a cell has multiple segments, each of which possibly connected to many neurons
        => Store feedforward and context in different sparse vectors

Active -> Predicted:
    This requires that we know which cells are active
        => Store a spvec of active cells
    


*/

typedef struct network_params_t_ {
    u32 num_minicols;
    u32 num_cells_per_minicol;
} network_params_t;

typedef struct inlayer_t_ {

    spvec_u8* feedforward_connections; // of shape (#Minicolumns, ...)
    spvec_u8** context_connections; // of shape (#Neurons, #Segments, ...)

    spvec_u1 predicted;
    spvec_u1* active_cells; // of shape (#minicols)

    network_params_t p;
} inlayer_t;

typedef struct outlayer_t_ {

    spvec_u8* feedforward_connections; // of shape (#Neurons, ...)
    spvec_u8** context_connections; // of shape (#Neurons, #Segments, ...)

    spvec_u1 predicted; 
    spvec_u1 active;

} outlayer_t;

void step(inlayer_t* n, dvec_u8 input, spvec_u1 active_columns);

void predict(u8* input, spvec_u1 active_columns);

void activate(u8* input, spvec_u1 active_columns);

#endif // NETWORK_H
