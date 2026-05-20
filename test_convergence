def test_gap_convergence(inst_metrics, plan_metrics):
    """
    Verify that adding planned future projects logically reduces the 
    capacity forecasting gap toward the Vision 2030 target.
    """
    # Unpack the 3rd element (index 2) which represents the 2030 capacity gap in Megawatts (MW)
    _, _, gap_installed, _, _ = inst_metrics
    _, _, gap_planned, _, _ = plan_metrics
    
    # Core Assertion: Including future project pipelines must mathematically shrink the remaining gap
    assert gap_planned < gap_installed, (
        f"Gap Convergence Failed: Planned track gap ({gap_planned:,.2f} MW) "
        f"should be strictly less than Installed baseline gap ({gap_installed:,.2f} MW)."
    )


def test_model_stability_convergence(inst_metrics, plan_metrics):
    """
    Check the mathematical bounds of the R² score to ensure data density 
    and model fitting converge reliably without chaotic standard errors.
    """
    # Unpack the 4th element (index 3) which contains the Coefficient of Determination (R² Score)
    _, _, _, r2_installed, _ = inst_metrics
    _, _, _, r2_combined, _ = plan_metrics
    
    # Mathematical Validation: R² scores must fundamentally fall within the bound of 0.0 to 1.0
    assert 0.0 <= r2_installed <= 1.0, f"Invalid R² score for Installed track: {r2_installed}"
    assert 0.0 <= r2_combined <= 1.0, f"Invalid R² score for Combined track: {r2_combined}"
    
    # Predictive Quality Threshold: Define a minimum acceptable variance standard (50% or greater)
    min_acceptable_r2 = 0.5
    
    # Guarantee that compiling historical + planned data yields a solid linear trend over pure noise
    assert r2_combined > min_acceptable_r2, (
        f"Model Convergence Error: Weak goodness-of-fit R² score ({r2_combined:.4f}). "
        f"Data variance is too high for reliable linear forecasting."
    )


def test_growth_rate_acceleration(inst_metrics, plan_metrics):
    """
    Verify that the estimated yearly growth rate (slope) accelerates 
    when pipelines shift from historical baselines to active upcoming plans.
    """
    # Unpack the 5th element (index 4) which contains the linear regression coefficient (Slope / MW per Year)
    _, _, _, _, slope_installed = inst_metrics
    _, _, _, _, slope_combined = plan_metrics
    
    # Policy Assertion: The strategic integration of upcoming plans must demonstrate a higher speed vector
    assert slope_combined > slope_installed, (
        f"Growth Rate Convergence Failed: Combined yearly additions ({slope_combined:,.2f} MW/year) "
        f"must show acceleration over current active historical speed ({slope_installed:,.2f} MW/year)."
    )


def run_all_convergence_tests(metrics_installed, metrics_planned):
    """
    Master controller function to orchestrate and execute the complete
    analytical convergence testing suite for the renewable data pipeline.
    """
    logging.info("Initializing analytical convergence suite...")
    try:
        # Step 1: Execute the gap reduction validity check
        test_gap_convergence(metrics_installed, metrics_planned)
        
        # Step 2: Validate the linear trend statistical metrics
        test_model_stability_convergence(metrics_installed, metrics_planned)
        
        # Step 3: Confirm acceleration in annual installation speed
        test_growth_rate_acceleration(metrics_installed, metrics_planned)
        
        # Print an formatted executive confirmation block upon complete success
        print("\n" + "=" * 60)
        print("ALL CONVERGENCE TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("  1. Gap Convergence        : Passed (Target gap is closing)")
        print("  2. Model Stability Fit    : Passed (R² scores are healthy & bounded)")
        print("  3. Growth Acceleration    : Passed (Future speed is accelerating)")
        print("=" * 60 + "\n")
        
        logging.info("All model and pipeline convergence tests passed successfully.")
        
    except AssertionError as e:
        # Capture precise mathematical failures thrown by the assert ecosystem
        logging.error(f"Convergence Test Failure: {e}")
        raise e
    except Exception as e:
        # Fallback block to capture completely unhandled operational errors (e.g., TypeErrors)
        logging.error(f"Unexpected error encountered during convergence test runtime: {e}")
        raise e
