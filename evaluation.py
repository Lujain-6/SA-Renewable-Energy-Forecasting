import logging
import warnings # to enable automated system alerts

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Evaluates system operational health and triggers early warnings if performance drift or data anomalies are detected.
def check_for_model_collapse(r2, estimated_yearly_growth):

    print(f"=== System Health Check ===")
    print(f"Current R² Score: {r2:.4f}")
    print(f"Estimated Growth: {estimated_yearly_growth:.2f} MW/year")
    print(f"===========================\n")

    # 1. Performance Check: Trigger warning if R² drops below standard threshold (0.5)
    if r2 < 0.5:
        warnings.warn(
            f"[CRITICAL WARNING] Potential Operational Drift Detected!\n"
            f"The R² score ({r2:.4f}) has dropped below the predictive quality threshold (0.5).\n"
            f"The model is publishing chaotic results or statistical noise.",
            UserWarning
        )
        
    # 2. Data Integrity Check: Trigger warning if estimated growth rate becomes negative
    if estimated_yearly_growth < 0:
        warnings.warn(
            f"[DATA INTEGRITY WARNING] Negative Yearly Growth Detected ({estimated_yearly_growth:.2f} MW/year).\n"
            f"Static CSV dataset might be outdated or real-world distribution changed.",
            UserWarning
        )
        
    return r2
    
def evaluate_dual_forecasts(inst_metrics, plan_metrics):
    try:
        print("\n" + "=" * 70)
        print("FINAL FORECAST EVALUATION REPORT")
        print("=" * 70)

        # Track 1 Parsing: Unpack Installed baseline metrics
        i_2030, i_target, i_gap, i_r2, i_slope = inst_metrics

        print("\nINSTALLED BASELINE ONLY TRACK (Current Actual Speed):")
        print(f"   - Cumulative Predicted 2030 Capacity : {i_2030:,.2f} MW")
        print(f"   - Vision 2030 Target                 : {i_target:,.2f} MW")
        print(f"   - 2030 Gap                           : {i_gap:,.2f} MW")
        print(f"   - R² Score                           : {i_r2:.4f}")
        print(f"   - Estimated Yearly Growth            : {i_slope:,.2f} MW/year")

        # Track 2 Parsing: Unpack Installed + Planned metrics
        p_2030, p_target, p_gap, p_r2, p_slope = plan_metrics

        print("\nCOMBINED INSTALLED + PLANNED TRACK:")
        print(f"   - Cumulative Predicted 2030 Capacity : {p_2030:,.2f} MW")
        print(f"   - Vision 2030 Target                 : {p_target:,.2f} MW")
        print(f"   - 2030 Gap                           : {p_gap:,.2f} MW")
        print(f"   - R² Score                           : {p_r2:.4f}")
        print(f"   - Estimated Yearly Growth            : {p_slope:,.2f} MW/year")

        # Clear and simple forecasting explanation for the user
        print("\nFORECAST INTERPRETATION:")
        print("   - The model uses historical renewable energy growth to estimate future cumulative capacity.")
        print("   - This forecast is approximate because it is based on limited historical data and a linear trend.")
        print(f"   - Using installed projects only, the 2030 capacity may be about {i_gap:,.0f} MW below the Vision 2030 target.")
        print(f"   - After adding planned projects, the expected 2030 gap decreases to about {p_gap:,.0f} MW.")
        print(f"   - Planned projects may reduce the gap by about {(i_gap - p_gap):,.0f} MW.")

        print("=" * 70)
        # Trigger the automated monitoring system for both tracks
        print("\n[Monitoring System Check - Installed Track]")
        check_for_model_collapse(i_r2, i_slope)
        
        print("[Monitoring System Check - Combined Track]")
        check_for_model_collapse(p_r2, p_slope)

    except Exception as e:
        logging.error(f"An unexpected runtime error occurred during final reporting phase: {e}")
        raise e
# evaluate function for app.py
def evaluate_forecast(future_2030, vision_target, gap_val, r2_val):
    try:
        achievement_rate = (future_2030 / vision_target) * 100 if vision_target > 0 else 0
        metrics = {
            'trend_fit': r2_val,
            'gap': gap_val if gap_val > 0 else 0,
            'achievement_rate': achievement_rate
        }
        return metrics
    except Exception as e:
        logging.error(f"Error in evaluate_forecast calculation for app: {e}")
        raise e
