# Evaluate predictive statistics against Vision targets with safety guards
def evaluate_dual_forecasts(inst_metrics, plan_metrics):
    try:
        logging.info("Initiating comprehensive comparative analysis on cumulative predictive tracks...")
        
        # Guard rail: Verify both positional arguments are structured as list/tuple matrices
        if not isinstance(inst_metrics, (tuple, list)) or not isinstance(plan_metrics, (tuple, list)):
            raise TypeError("Track evaluation inputs must be structured inside a tuple or list package.")

        print("\n" + "="*70)
        print("          RENEWABLE ENERGY COMPREHENSIVE FORECAST EVALUATION          ")
        print("="*70)
        
        # Track 1 Parsing: Unpack historical baseline metrics
        i_2030, i_target, i_r2, i_slope = inst_metrics
        print(f"1-INSTALLED BASELINE ONLY TRACK (Current Actual Speed):")
        print(f"    - Cumulative Predicted 2030 Capacity : {i_2030:,.2f} MW")
        print(f"    - Strategic Vision 2030 Target        : {i_target:,.2f} MW")
        print(f"    - Target Shortfall Remaining Gap      : {i_target - i_2030:,.2f} MW")
        print(f"    - Model Statistical Accuracy Score R²: {i_r2:.4f}")
        print(f"    - Historical Annualized Growth Speed  : {i_slope:,.2f} MW/year")
        
        print("-"*70)
        
        # Track 2 Parsing: Unpack combined future pipeline metrics
        p_2030, p_target, p_r2, p_slope = plan_metrics
        print(f"2-COMBINED TARGET TRACK (Installed Baseline + Planned Pipeline):")
        print(f"    - Cumulative Predicted 2030 Capacity : {p_2030:,.2f} MW")  
        print(f"    - Strategic Vision 2030 Target        : {p_target:,.2f} MW")
        
        # Dynamic context checking to compute gaps correctly even if the prediction overshoots the target
        gap = p_target - p_2030
        if gap > 0:
            print(f"    - Expected Deficit Gap to Target      : {gap:,.2f} MW")
        else:
            print(f"    - Target Achieved! Expected Surplus   : {abs(gap):,.2f} MW")
            
        print(f"    - Model Statistical Accuracy Score R²: {p_r2:.4f}")
        print(f"    - Accelerated Combined Growth Speed   : {p_slope:,.2f} MW/year")  
        
        logging.info("Comprehensive analytical scoring matrix compiled and reported successfully.")
        
        # Return is placed at the absolute end of successful execution block
        return True

    except TypeError as e:
        logging.error(f"Type structure failure within evaluate_dual_forecasts: {e}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected runtime error occurred during final reporting phase: {e}")
        raise e
