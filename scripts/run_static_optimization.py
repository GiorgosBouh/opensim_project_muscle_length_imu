import sys
import os
import opensim as osim

def run_static_optimization(
    model_path,
    mot_path,
    results_dir,
    start_time=None,
    end_time=None
):
    # Έλεγχοι αρχείων
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(mot_path):
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    # Δημιουργία φακέλου αποτελεσμάτων
    os.makedirs(results_dir, exist_ok=True)

    # Βασικά ονόματα
    mot_base = os.path.splitext(os.path.basename(mot_path))[0]
    prefix = mot_base + "_SO"
    # OpenSim by default θα βγάλει αρχεία τύπου:
    #   <prefix>_StaticOptimization_force.sto
    # στον φάκελο results_dir

    print(f"📄 Model: {model_path}")
    print(f"📄 MOT:   {mot_path}")
    print(f"📂 Results dir: {results_dir}")
    print(f"🔧 Prefix: {prefix}")

    # Φορτώνουμε το μοντέλο
    model = osim.Model(model_path)
    state = model.initSystem()

    # Ανιχνεύουμε start/end time από το .mot αν δεν δοθούν
    if start_time is None or end_time is None:
        table = osim.TimeSeriesTable(mot_path)
        times = table.getIndependentColumn()
        t0 = float(times[0])
        t1 = float(times[-1])
        if start_time is None:
            start_time = t0
        if end_time is None:
            end_time = t1

    print(f"⏱ Time range: {start_time:.3f} – {end_time:.3f} s")

    # Φτιάχνουμε StaticOptimization analysis
    so = osim.StaticOptimization()
    so.setName("StaticOptimization")
    so.setStartTime(start_time)
    so.setEndTime(end_time)
    # optional: exponent for cost function (2 = sum of squared activations)
    so.setActivationExponent(2)

    # Προσθέτουμε το analysis στο μοντέλο
    model.addAnalysis(so)

    # Configure AnalyzeTool
    analyze_tool = osim.AnalyzeTool()
    analyze_tool.setModel(model)
    analyze_tool.setName(prefix)
    analyze_tool.setCoordinatesFileName(mot_path)
    analyze_tool.setInitialTime(start_time)
    analyze_tool.setFinalTime(end_time)
    analyze_tool.setLowpassCutoffFrequency(6.0)  # optional smoothing
    analyze_tool.setLoadModelAndInput(True)
    analyze_tool.setResultsDir(results_dir)

    print("🚀 Running Static Optimization via AnalyzeTool...")
    analyze_tool.run()
    print("✅ Static Optimization completed.")

    print("\nℹ Expected output:")
    print(f"   {os.path.join(results_dir, prefix + '_StaticOptimization_force.sto')}")
    print(f"   {os.path.join(results_dir, prefix + '_StaticOptimization_activation.sto')} (αν υπάρχει)")

def main():
    if len(sys.argv) < 4:
        print("Usage: python run_static_optimization.py <model.osim> <input.mot> <results_dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    mot_path = sys.argv[2]
    results_dir = sys.argv[3]

    run_static_optimization(model_path, mot_path, results_dir)

if __name__ == "__main__":
    main()
