import os

from tqdm import tqdm

from .heuristic import HeuristicSolver
from .max_step_sr import MaxStepSRSolver
from .multi_step_sr import MultiStepSRSolver
from .oblivious_routing import ObliviousRoutingSolver
from .one_step_sr import OneStepSRSolver
from .util import *
from .util_h import count_routing_change


def get_route_changes(routings, G):
    route_changes = np.zeros(shape=(routings.shape[0] - 1))
    for t in range(routings.shape[0] - 1):
        _route_changes = 0
        for i, j in itertools.product(range(routings.shape[1]), range(routings.shape[2])):
            path_t_1 = get_paths(G, routings[t + 1], i, j)
            path_t = get_paths(G, routings[t], i, j)
            if path_t_1 != path_t:
                _route_changes += 1

        route_changes[t] = _route_changes

    return route_changes


def get_route_changes_heuristic(routings):
    route_changes = []
    for t in range(routings.shape[0] - 1):
        route_changes.append(count_routing_change(routings[t + 1], routings[t]))

    route_changes = np.asarray(route_changes)
    return route_changes


def extract_results(results):
    mlus, solutions = [], []
    for _mlu, _solution in results:
        mlus.append(_mlu)
        solutions.append(_solution)

    mlus = np.stack(mlus, axis=0)
    solutions = np.stack(solutions, axis=0)

    return mlus, solutions


def save_results(log_dir, fname, mlus, route_change):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    np.save(os.path.join(log_dir, fname + '_mlus'), mlus)
    np.save(os.path.join(log_dir, fname + '_route_change'), route_change)


def get_te_data(x_gt, y_gt, yhat, args):
    te_step = args.test_size if args.te_step is 0 else args.te_step
    nsteps = len(range(0, te_step, args.seq_len_y))

    if x_gt.shape[0] > nsteps * 2:
        x_gt = x_gt[0:te_step:args.seq_len_y]
        y_gt = y_gt[0:te_step:args.seq_len_y]
        yhat = yhat[0:te_step:args.seq_len_y]

    return x_gt, y_gt, yhat


def solve_oblivious_routing(yhat, y_gt, x_gt, G, segments, te_step, args):
    oblivious_solver = ObliviousRoutingSolver(G, segments)
    oblivious_solver.solve()
    print('Solving Obilious Routing: Done')
    oblivious_results = []
    for i in tqdm(range(te_step)):
        oblivious_results.append(do_te(c='oblivious', tms=y_gt[i], gt_tms=y_gt[i], G=G,
                                       last_tm=np.max(x_gt[i], axis=0), nNodes=args.nNodes, solver=oblivious_solver))

    mlu_oblivious, solutions_oblivious = extract_results(oblivious_results)
    route_changes_or = get_route_changes(solutions_oblivious, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_or), np.std(route_changes_or)))

    print('Oblivious              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_oblivious),
                                                                              np.mean(mlu_oblivious),
                                                                              np.max(mlu_oblivious),
                                                                              np.std(mlu_oblivious)))
    save_results(args.log_dir, 'oblivious', mlu_oblivious, route_changes_or)

    pred_results = Parallel(n_jobs=os.cpu_count() - 2)(delayed(do_te)(
        c=args.type, tms=yhat[i], gt_tms=y_gt[i], G=G,
        last_tm=np.max(x_gt[i], axis=0), nNodes=args.nNodes) for i in range(te_step))

    mlu_pred, solutions_pred = extract_results(pred_results)
    route_changes_pred = get_route_changes(solutions_pred, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_pred), np.std(route_changes_pred)))
    print('{} {}               | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.type, args.model,
                                                                           np.min(mlu_pred),
                                                                           np.mean(mlu_pred),
                                                                           np.max(mlu_pred),
                                                                           np.std(mlu_pred)))

    save_results(args.log_dir, '{}_{}'.format(args.model, args.type), mlu_pred, route_changes_pred)


def last_step_solver(yhat, y_gt, x_gt, G, segments, te_step, args):
    one_step_pred_results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='last_step', tms=yhat[i], gt_tms=y_gt[i], G=G,
        last_tm=yhat[i][0], nNodes=args.nNodes) for i in range(te_step))

    mlu_one_step_pred, solutions_one_step_pred = extract_results(one_step_pred_results)
    route_changes_one_step_pred = get_route_changes(solutions_one_step_pred, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes_one_step_pred),
                                                        np.std(route_changes_one_step_pred)))
    print('Ones-step prediction    | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
        np.min(mlu_one_step_pred),
        np.mean(mlu_one_step_pred),
        np.max(mlu_one_step_pred),
        np.std(mlu_one_step_pred)))

    save_results(args.log_dir, 'one_step_pred_{}'.format(args.model), mlu_one_step_pred, route_changes_one_step_pred)

    results_last_step = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='last_step', tms=y_gt[i], gt_tms=y_gt[i], G=G,
        last_tm=x_gt[i][-1], nNodes=args.nNodes) for i in range(te_step))

    mlu_last_step, solution_last_step = extract_results(results_last_step)
    route_changes_last_step = get_route_changes(solution_last_step, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_last_step),
                                                        np.std(route_changes_last_step)))
    print('last-step            | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_last_step),
                                                                            np.mean(mlu_last_step),
                                                                            np.max(mlu_last_step),
                                                                            np.std(mlu_last_step)))

    save_results(args.log_dir, 'last_step', mlu_last_step, route_changes_last_step)


def one_step_pred_solver(yhat, y_gt, x_gt, G, segments, te_step, args):
    one_step_pred_results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='last_step', tms=yhat[i], gt_tms=y_gt[i], G=G,
        last_tm=yhat[i][0], nNodes=args.nNodes) for i in range(te_step))

    mlu_one_step_pred, solutions_one_step_pred = extract_results(one_step_pred_results)
    route_changes_one_step_pred = get_route_changes(solutions_one_step_pred, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes_one_step_pred),
                                                        np.std(route_changes_one_step_pred)))
    print('Ones-step prediction    | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(
        np.min(mlu_one_step_pred),
        np.mean(mlu_one_step_pred),
        np.max(mlu_one_step_pred),
        np.std(mlu_one_step_pred)))

    save_results(args.log_dir, 'one_step_pred_heiristic_{}'.format(args.model), mlu_one_step_pred,
                 route_changes_one_step_pred)

    results_last_step = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='last_step', tms=y_gt[i], gt_tms=y_gt[i], G=G,
        last_tm=x_gt[i][-1], nNodes=args.nNodes) for i in range(te_step))

    mlu_last_step, solution_last_step = extract_results(results_last_step)
    route_changes_last_step = get_route_changes(solution_last_step, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_last_step),
                                                        np.std(route_changes_last_step)))
    print('last-step            | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_last_step),
                                                                            np.mean(mlu_last_step),
                                                                            np.max(mlu_last_step),
                                                                            np.std(mlu_last_step)))

    save_results(args.log_dir, 'last_step', mlu_last_step, route_changes_last_step)


def optimal_p1_solver(yhat, y_gt, x_gt, G, segments, te_step, args):
    results_optomal_p1 = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='p1', tms=y_gt[i], gt_tms=y_gt[i], G=G,
        last_tm=np.max(x_gt[i], axis=0), nNodes=args.nNodes) for i in range(te_step))

    mlu_optimal_p1, solution_optimal_p1 = extract_results(results_optomal_p1)
    route_changes_p1 = get_route_changes(solution_optimal_p1, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_p1), np.std(route_changes_p1)))
    print('P1                   | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_optimal_p1),
                                                                            np.mean(mlu_optimal_p1),
                                                                            np.max(mlu_optimal_p1),
                                                                            np.std(mlu_optimal_p1)))

    save_results(args.log_dir, 'p1_optimal', mlu_optimal_p1, route_changes_p1)

    results_optimal_p2 = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='p2', tms=np.max(y_gt[i], axis=0, keepdims=True), gt_tms=y_gt[i], G=G,
        last_tm=np.max(x_gt[i], axis=0, keepdims=True), nNodes=args.nNodes) for i in range(te_step))

    mlu_optimal_p2, solution_optimal_p2 = extract_results(results_optimal_p2)
    route_changes_p2 = get_route_changes(solution_optimal_p2, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_p2), np.std(route_changes_p2)))
    print('P2                   | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_optimal_p2),
                                                                            np.mean(mlu_optimal_p2),
                                                                            np.max(mlu_optimal_p2),
                                                                            np.std(mlu_optimal_p2)))

    save_results(args.log_dir, 'p2_optimal', mlu_optimal_p2, route_changes_p2)


def ls2sr(yhat, y_gt, x_gt, G, segments, te_step, args):
    print('ls2sr solver')
    results = []
    solver = HeuristicSolver(G, time_limit=1, verbose=args.verbose)

    solution = None
    for i in range(te_step):
        mean = np.mean(y_gt[i], axis=1)
        std_mean = np.std(mean)
        var = np.var(y_gt[i], axis=1)
        std_var = np.std(var)
        u, solution = p2_heuristic_solver(solver, tm=yhat[i],
                                          gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)
        print(np.sum(y_gt[i]), ' ', std_mean, ' ', std_var, ' ', np.mean(u))
        results.append((u, solution))

    mlu, solution = extract_results(results)
    route_changes = get_route_changes_heuristic(solution)
    print(
        'Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                      (args.seq_len_y * route_changes.shape[0]),
                                                      np.std(route_changes)))
    print('P2 Heuristic    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                               np.min(mlu),
                                                                               np.mean(mlu),
                                                                               np.max(mlu),
                                                                               np.std(mlu)))

    save_results(args.log_dir, 'p2_heuristic', mlu, route_changes)


def optimal_p3_solver(yhat, y_gt, x_gt, G, segments, te_step, args):
    t_prime = int(args.seq_len_y / args.trunk)
    results_optimal_p3 = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='p3', tms=np.stack([np.max(y_gt[i][j:j + t_prime], axis=0) for j in range(0, y_gt[i].shape[0], t_prime)]),
        gt_tms=y_gt[i], G=G, last_tm=np.max(x_gt[i], axis=0), nNodes=args.nNodes) for i in
                                                             range(te_step))

    mlu_optimal_p3, solution_optimal_p3 = extract_results(results_optimal_p3)
    route_changes_p3 = get_route_changes(solution_optimal_p3, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_p3), np.std(route_changes_p3)))
    print('P3                   | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_optimal_p3),
                                                                            np.mean(mlu_optimal_p3),
                                                                            np.max(mlu_optimal_p3),
                                                                            np.std(mlu_optimal_p3)))

    save_results(args.log_dir, 'p3_optimal', mlu_optimal_p3, route_changes_p3)


def optimal_solver(yhat, y_gt, x_gt, G, segments, te_step, args):
    results_optimal = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='optimal', tms=y_gt[i], gt_tms=y_gt[i], G=G,
        last_tm=np.max(x_gt[i], axis=0), nNodes=args.nNodes) for i in range(te_step))

    mlu_optimal, solution_optimal = extract_results(results_optimal)
    solution_optimal = np.reshape(solution_optimal, newshape=(-1, args.nNodes, args.nNodes, args.nNodes))
    route_changes_opt = get_route_changes(solution_optimal, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_opt), np.std(route_changes_opt)))
    print('Optimal              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_optimal),
                                                                            np.mean(mlu_optimal),
                                                                            np.max(mlu_optimal),
                                                                            np.std(mlu_optimal)))
    save_results(args.log_dir, 'optimal_optimal', mlu_optimal, route_changes_opt)


def ls2sr_p0_solver(yhat, y_gt, x_gt, G, segments, te_step, args):
    print('P0 Heuristic solver')
    solver_pred_p0_heuristic = HeuristicSolver(G, time_limit=10, verbose=args.verbose)

    results_pred_p0_heuristic = Parallel(n_jobs=os.cpu_count() - 8)(delayed(p0_heuristic_solver)(
        solver=solver_pred_p0_heuristic, tms=y_gt[i], gt_tms=y_gt[i], p_solution=None, nNodes=args.nNodes)
                                                                    for i in range(te_step))

    mlu_pred_p0_heuristic, solution_pred_p0_heuristic = extract_results(results_pred_p0_heuristic)
    route_changes_pred_p0_heuristic = get_route_changes_heuristic(solution_pred_p0_heuristic)
    print(
        'Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_pred_p0_heuristic),
                                                      np.std(route_changes_pred_p0_heuristic)))
    print('P0 Heuristic    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                               np.min(mlu_pred_p0_heuristic),
                                                                               np.mean(mlu_pred_p0_heuristic),
                                                                               np.max(mlu_pred_p0_heuristic),
                                                                               np.std(mlu_pred_p0_heuristic)))

    save_results(args.log_dir, 'p0_heuristic', mlu_pred_p0_heuristic, route_changes_pred_p0_heuristic)


def run_te(x_gt, y_gt, yhat, args):
    G = load_network_topology(args.dataset)

    if not os.path.isfile('../../data/topo/{}_segments.npy'.format(args.dataset)):

        segments = get_segments(G)
        np.save('../../data/topo/{}_segments'.format(args.dataset), segments)
    else:
        segments = np.load('../../data/topo/{}_segments.npy'.format(args.dataset), allow_pickle=True)

    x_gt, y_gt, yhat = get_te_data(x_gt, y_gt, yhat, args)

    te_step = x_gt.shape[0]
    print('    Method           |   Min     Avg    Max     std')

    ls2sr(yhat, y_gt, x_gt, G, segments, te_step, args)


def do_te(c, tms, gt_tms, G, last_tm, nNodes=12, solver_type='pulp_coin', solver=None):
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0
    last_tm[last_tm <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))

    segments = get_segments(G)

    if c == 'p1':
        multi_step_solver = MultiStepSRSolver(G, segments)
        return multi_step_sr(multi_step_solver, tms, gt_tms)
    elif c == 'p3':
        mms_solver = MultiStepSRSolver(G, segments)
        return mms_sr(mms_solver, tms, gt_tms)
    elif c == 'last_step':
        last_tm = last_tm.reshape((nNodes, nNodes))
        last_tm = last_tm * (1.0 - np.eye(nNodes))

        last_step_solver = OneStepSRSolver(G, segments)
        return last_step_sr(last_step_solver, last_tm, gt_tms)
    elif c == 'p2':
        tms = tms.reshape((nNodes, nNodes))

        if solver_type == 'pulp_coin':
            max_step_solver = MaxStepSRSolver(G, segments)
        else:
            raise NotImplementedError('Solver not implemented!')

        return max_step_sr(max_step_solver, tms, gt_tms)
    elif c == 'optimal':
        optimal_solver = OneStepSRSolver(G, segments)
        return optimal_sr(optimal_solver, gt_tms)
    elif c == 'oblivious':
        if solver is not None:
            oblivious_solver = solver
        else:
            oblivious_solver = ObliviousRoutingSolver(G, segments)
        return oblivious_sr(oblivious_solver, tms)
    else:
        raise ValueError('TE not found')


def multi_step_sr(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(get_max_utilization_v2(solver, gt_tms[i]))
    return u, solver.solution


def mms_sr(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(get_max_utilization_v2(solver, gt_tms[i]))
    return u, solver.solution


def max_step_sr(solver, tms, gt_tms):
    u = []

    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(get_max_utilization_v2(solver, gt_tms[i]))
    return u, solver.solution


def max_step_ls2sr(solver, tm, gt_tms, path_solution, nNodes):
    u = []
    tm = tm.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tm[tm <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tm[:] = tm[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tm = tm.reshape((nNodes, nNodes))

    # _s = time.time()
    solution, path_solution = solver.solve_v2(tm, solution=path_solution)  # solve backtrack solution (line 131)

    # print('solving time: ', time.time() - _s)
    for i in range(gt_tms.shape[0]):
        u.append(get_max_utilization_v2(solver, gt_tms[i]))
    return u, solution, path_solution


def p0_heuristic_solver(solver, tms, gt_tms, p_solution, nNodes):
    u = []
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tms = tms.reshape((-1, nNodes, nNodes))

    solutions = []
    for i in range(gt_tms.shape[0]):
        solution = solver.solve(tms[i], solution=p_solution)  # solve backtrack solution (line 131)
        u.append(solver.evaluate(solution, gt_tms[i]))
        solutions.append(solution)

    return u, solutions


def p2_heuristic_solver(solver, tm, gt_tms, p_solution, nNodes):
    u = []
    tm = tm.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tm[tm <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tm[:] = tm[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tm = tm.reshape((nNodes, nNodes))

    # _s = time.time()
    try:
        solution = solver.solve(tm, solution=p_solution)  # solve backtrack solution (line 131)
    except:
        solution = solver.initialize()
    # print('solving time: ', time.time() - _s)
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(solution, gt_tms[i]))
    return u, solution


def last_step_sr(solver, last_tm, gt_tms):
    u = []
    try:
        solver.solve(last_tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(get_max_utilization_v2(solver, gt_tms[i]))
    return u, solver.solution


def optimal_sr(solver, gt_tms):
    u = []
    solutions = []
    for i in range(gt_tms.shape[0]):
        try:
            solver.solve(gt_tms[i])
        except:
            pass
        solutions.append(solver.solution)
        u.append(get_max_utilization_v2(solver, gt_tms[i]))

    solutions = np.stack(solutions, axis=0)
    return u, solutions


def oblivious_sr(solver, tms):
    u = []
    for i in range(tms.shape[0]):
        u.append(get_max_utilization_v2(solver, tms[i]))

    return u, solver.solution
