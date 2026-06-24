def maximum_path_c(paths, values, t_ys, t_xs):
    '''Pure-Python fallback for monotonic alignment dynamic programming.'''
    import numpy as np

    bsz = values.shape[0]
    for b in range(bsz):
        ty = int(t_ys[b])
        tx = int(t_xs[b])
        if ty <= 0 or tx <= 0:
            continue

        v = values[b, :ty, :tx]
        dp = np.full((ty, tx), -1e30, dtype=np.float32)
        ptr = np.zeros((ty, tx), dtype=np.int8)
        dp[0, 0] = v[0, 0]

        for x in range(1, tx):
            dp[0, x] = dp[0, x - 1] + v[0, x]

        for y in range(1, ty):
            # At least y x-steps are needed to reach y, so start at x=y.
            for x in range(y, tx):
                stay = dp[y, x - 1] if x > 0 else -1e30
                advance = dp[y - 1, x - 1] if x > 0 else -1e30
                if advance >= stay:
                    dp[y, x] = advance + v[y, x]
                    ptr[y, x] = 1
                else:
                    dp[y, x] = stay + v[y, x]

        y = ty - 1
        for x in range(tx - 1, -1, -1):
            paths[b, y, x] = 1
            if y > 0 and ptr[y, x] == 1:
                y -= 1
