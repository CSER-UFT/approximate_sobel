#ifndef COMPONENTS_H
#define COMPONENTS_H

#include "templates/add_exact.h"
#include "templates/add_trunc0_1.h"
#include "templates/add_trunc0_2.h"
#include "templates/add_trunc0_3.h"
#include "templates/add_trunc0_4.h"
#include "templates/add_trunc0_5.h"
#include "templates/add_trunc1_1.h"
#include "templates/add_trunc1_2.h"
#include "templates/add_trunc1_3.h"
#include "templates/add_trunc1_4.h"
#include "templates/add_trunc1_5.h"

typedef uint12_t_hw (*adder_func)(uint12_t_hw, uint12_t_hw);

static const adder_func add[11] = {
    add_exact,
    add_trunc0_1, add_trunc0_2, add_trunc0_3, add_trunc0_4, add_trunc0_5,
    add_trunc1_1, add_trunc1_2, add_trunc1_3, add_trunc1_4, add_trunc1_5
};

#endif // COMPONENTS_H
