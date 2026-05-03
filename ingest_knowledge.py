from docling.document_converter import DocumentConverter
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
SOURCE_PDFS = [
    "knowledge_base/basics_of_strength_and_conditioning_manual.pdf",
    "knowledge_base/american_college_of_sports_medicine_position.pdf",
    "knowledge_base/an_analysis_of_teaching_and_coaching_behaviors.pdf",
    "knowledge_base/basics_of_training_for_muscle_size_theory.pdf",
    "knowledge_base/designing_weight_training_programs.pdf",
    "knowledge_base/determinants_of_olympic_fencing_performance.pdf",
    "knowledge_base/effects_of_betaine_supplementation.pdf",
    "knowledge_base/effects_of_creatine.pdf",
    "knowledge_base/effects_of_low_vs_high_load_resistance_training.pdf",
    "knowledge_base/effects_of_rest_intervals.pdf",
    "knowledge_base/effects_of_training_on_muscle_cells.pdf",
    "knowledge_base/free_weight_resistance_training_youth.pdf",
    "knowledge_base/fundamentals_of_resistance_training.pdf",
    "knowledge_base/Golf_and_Physical_Health_A_Systematic_Review.pdf",
    "knowledge_base/periodized_resistance_training_and_strength.pdf",
    "knowledge_base/resistance_training_among_young_athletes.pdf",
    "knowledge_base/resistance_training_and_elite_athletes_adaptations.pdf",
    "knowledge_base/resistance_training_for_older_adults.pdf",
    "knowledge_base/Resistance_Training_is_Medicine.pdf",
    "knowledge_base/resistance_training_on_swimming.pdf",
    "knowledge_base/squatting_kinematics_and_kinetics.pdf",
    "knowledge_base/the_effect_of_training_at_a_specific_time_of_day.pdf",
    "knowledge_base/the_effects_of_strength_and_conditioning_on_sprinting.pdf",
    "knowledge_base/the_mechanisms_of_muscle_hypertrophy.pdf",
    "knowledge_base/the_perceived_psychological_responsibilities_of_a_coach.pdf",
    "knowledge_base/times_per_week_training_hypertrophy.pdf",
    "knowledge_base/total_number_of_sets_for_hypertrophy.pdf",
    # Add more paths here as needed
]
OUTPUT_DIR = "data/knowledge_markdown"


def convert_pdf_to_markdown(source_pdf: str, converter: DocumentConverter):
    """
    Convert a single PDF to a Markdown file using a shared converter instance.
    Skips conversion if the output file already exists.
    """

    if not os.path.exists(source_pdf):
        print(f"❌ File not found: {source_pdf}")
        return None

    # Derive output filename
    base_name = os.path.splitext(os.path.basename(source_pdf))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")

    # ── SKIP LOGIC ──
    if os.path.exists(output_path):
        print(f"⏩ Skipping: {base_name}.md already exists.")
        return output_path

    print(f"🚀 Docling is analyzing: {source_pdf}")

    try:
        # Perform the conversion
        result = converter.convert(source_pdf)
        markdown_output = result.document.export_to_markdown()

        # Ensure directory exists and write file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)

        size_kb = len(markdown_output.encode("utf-8")) / 1024
        print(f"✅ Saved to {output_path} ({size_kb:.1f} KB extracted)")
        return output_path

    except Exception as e:
        print(f"❌ Conversion failed for {source_pdf}: {e}")
        return None


def run():
    print(f"Checking {len(SOURCE_PDFS)} PDF(s)...\n")

    # Initialize the converter ONCE here instead of inside the function.
    # This prevents the script from reloading models for every file.
    converter = DocumentConverter()

    success, failed = 0, 0

    for pdf in SOURCE_PDFS:
        result = convert_pdf_to_markdown(pdf, converter)
        if result:
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Done. ✅ {success} processed (new or skipped)  ❌ {failed} failed")
    if success > 0:
        print(f"Run build_vector_db.py next to index the markdown files.")


if __name__ == "__main__":
    run()