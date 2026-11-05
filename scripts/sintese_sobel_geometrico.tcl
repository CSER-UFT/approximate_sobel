# ==============================================================================
# sintese_sobel.tcl - Automação da Síntese Vitis HLS para o Filtro Sobel
# Dispositivo: Xilinx Spartan-7 (xc7s50csga324-1) | Clock: 10 ns (100 MHz)
# ==============================================================================

set PART "xc7s50csga324-1"
set CLK_PERIOD 10
set REPORTS_DIR "reports_sintese_geometrico"
file mkdir $REPORTS_DIR

set configs {
    {"th_00" "src_sintese_geometrico/sobel_th_00.cpp" "proj_sobel_th_00" "Baseline (Tudo Exato)"}
    {"th_05" "src_sintese_geometrico/sobel_th_05.cpp" "proj_sobel_th_05" "Threshold 5%"}
    {"th_10" "src_sintese_geometrico/sobel_th_10.cpp" "proj_sobel_th_10" "Threshold 10%"}
    {"th_15" "src_sintese_geometrico/sobel_th_15.cpp" "proj_sobel_th_15" "Threshold 15%"}
    {"th_20" "src_sintese_geometrico/sobel_th_20.cpp" "proj_sobel_th_20" "Threshold 20%"}
    {"th_25" "src_sintese_geometrico/sobel_th_25.cpp" "proj_sobel_th_25" "Threshold 25%"}
    {"th_30" "src_sintese_geometrico/sobel_th_30.cpp" "proj_sobel_th_30" "Threshold 30%"}
}

foreach cfg $configs {
    set tag   [lindex $cfg 0]
    set src   [lindex $cfg 1]
    set proj  [lindex $cfg 2]
    set desc  [lindex $cfg 3]

    puts "\n=================================================================="
    puts " === VITIS HLS: SINTETIZANDO $desc ($tag) ==="
    puts " Arquivo: $src | Projeto: $proj"
    puts "=================================================================="

    open_project $proj -reset
    add_files $src -cflags "-I. -Itemplates"
    set_top sobel

    open_solution "solution1" -reset
    set_part $PART
    create_clock -period $CLK_PERIOD -name default

    puts "--- Executando C-Synthesis (csynth_design) ---"
    csynth_design

    set rpt_file "$proj/solution1/syn/report/sobel_csynth.rpt"
    set xml_file "$proj/solution1/syn/report/sobel_csynth.xml"
    if {[file exists $rpt_file]} {
        file copy -force $rpt_file "$REPORTS_DIR/${tag}_csynth.rpt"
        puts "✔ Relatório salvo em: $REPORTS_DIR/${tag}_csynth.rpt"
    }
    if {[file exists $xml_file]} {
        file copy -force $xml_file "$REPORTS_DIR/${tag}_csynth.xml"
        puts "✔ Relatório XML salvo em: $REPORTS_DIR/${tag}_csynth.xml"
    }

    close_project
}

puts "\n=================================================================="
puts "✔ SÍNTESE CONCLUÍDA PARA TODOS OS DESIGNS (BASELINE + 5% A 30%)!"
puts "==================================================================\n"
exit
