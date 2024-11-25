#include "network.h"

void step(inlayer_t* n, dvec_u8 input, spvec_u1 active_columns) {
    // predicted -> active
    for(u32 it = 0; it < active_columns.non_null_count; ++it) {
        u32 col_it = active_columns.indices[it];
        for(u32 cell_it = 0; cell_it < n->p.num_cells_per_minicol; ++cell_it) {
            
        }
    }


    // active -> predicted
}
