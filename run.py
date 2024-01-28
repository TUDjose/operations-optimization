import Model.model as model
import Model.data_generator as dg
import Model.sensitivity as sa
import argparse

def runner():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--sens', action='store_true')

    parser.add_argument('-file', type=str, default='tests/test_data.csv')
    parser.add_argument('-u', type=int, default=4)
    parser.add_argument('-qc', type=int, default=2500)
    parser.add_argument('-capacity', type=int, default=300)
    parser.add_argument('-n', type=int, default=4)
    parser.add_argument('-m', type=int, default=5)
    parser.add_argument('-fq', type=float, default=1.2)

    args = parser.parse_args()

    if not args.gen and not args.sens:
        func_to_run = getattr(model, 'run_model', None)
        if func_to_run and callable(func_to_run):
            func_to_run(args.file, args.u, args.qc, args.capacity)

    if args.gen:
        data = dg.DataGenerator(args.m, args.n, args.u, args.fq)
        data.store()

    if args.sens:
        print(sa.w_sensitivity()[0])
        print(sa.p_sensitivity())


if __name__ == '__main__':
    runner()