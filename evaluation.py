import logging

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
