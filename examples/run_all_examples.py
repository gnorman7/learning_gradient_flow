import ex1_2vars
import ex5_deep
import examples.ex3_newton as ex3_newton
import ex4_scattering
from example_config import load_config


def run_all_examples(threshold: float, alpha: float):
    results = {}

    print("Running ex1_2vars...")
    results["ex1_2vars"] = ex1_2vars.run_example(threshold=threshold, alpha=alpha)

    print("Running ex5_deep...")
    results["ex5_deep"] = ex5_deep.run_example(threshold=threshold, alpha=alpha)

    print("Running ex2_newton...")
    results["ex2_newton"] = ex3_newton.run_example(threshold=threshold, alpha=alpha)

    # Note that for the scattering we set alpha=0.0 here.
    # print("Running ex4_scattering...")
    # results["ex4_scattering"] = ex4_scattering.run_example(threshold=threshold, alpha=0.0)

    return results


if __name__ == "__main__":
    config = load_config()
    run_all_examples(threshold=config["threshold"], alpha=config["alpha"])
