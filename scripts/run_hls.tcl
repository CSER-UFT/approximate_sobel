open_project proj_somadores -reset

add_files src/add_exact.cpp
add_files src/add_trunc0_1.cpp
add_files src/add_trunc0_2.cpp
add_files src/add_trunc0_3.cpp
add_files src/add_trunc0_4.cpp
add_files src/add_trunc0_5.cpp
add_files src/add_trunc1_1.cpp
add_files src/add_trunc1_2.cpp
add_files src/add_trunc1_3.cpp
add_files src/add_trunc1_4.cpp
add_files src/add_trunc1_5.cpp

add_files -tb tb_main.cpp

set meus_somadores {
    add_exact
    add_trunc0_1
    add_trunc0_2
    add_trunc0_3
    add_trunc0_4
    add_trunc0_5
    add_trunc1_1
    add_trunc1_2
    add_trunc1_3
    add_trunc1_4
    add_trunc1_5
}

# 4. Laço de Repetição para rodar um a um
foreach modulo $meus_somadores {

    set_top $modulo    
    open_solution "sol_$modulo" -reset
    set_part {xc7s50csga324-1}
    create_clock -period 10 -name default
    
    puts "================================================="
    puts " INICIANDO SINTESE DO MODULO: $modulo"
    puts "================================================="
    
    csynth_design
}

exit
