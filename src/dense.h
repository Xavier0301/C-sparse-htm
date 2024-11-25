#ifndef DENSE_H
#define DENSE_H

#include "types.h"
#include "tensor.h"

typedef struct dvec_u8_ {
        u16 length;
        u8* data;
} dvec_u8;

typedef struct minicol_network_t_ {

    mat_u8 feedforward_connections; // of shape (#minicolumns, #neurons_per)
    tensor_u8 context_connections; // of shape (#neurons, #segments, neurons)

    dvec_u8* feedforward_connections; // of shape (#Minicolumns, ...)
    dvec_u8** context_connections; // of shape (#Neurons, #Segments, ...)

    dvec_u8 predicted;
    dvec_u8 active_minicols;

} minicol_network_t;



#endif // DENSE_H
