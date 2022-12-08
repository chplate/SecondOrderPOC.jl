function getProblem(index)
  lb = SecondOrderPOC.emptyfvec
  ub = SecondOrderPOC.emptyfvec

  if index == 1
      name = "DoubleTank"
      t0 = 0.0
      tf = 10.0

      C = [0.0 1.0 -1.0;
           0.0 1.0 -1.0 ]
      Q = C'*C
      x0 = [2.0; 2.0; 3.0]

      n_omega = 2

      function funDoubleTank(x)
        f = zeros(eltype(x), 3,3)
        f[1,1] = -sqrt(x[1])
        f[2,1] = sqrt(x[1]) - sqrt(x[2])

        f[1,2] = 1

        f[1,3] = 2
        return f

      end

    return name, t0, tf, x0, Q, funDoubleTank, lb, ub, n_omega


  elseif index == 2
      name = "DoubleTankMM"
      t0 = 0.0
      tf = 10.0

      C = [0.0 1.0 -1.0;
           0.0 1.0 -1.0 ]
      Q = C'*C
      x0 = [2.0; 2.0; 3.0]

      n_omega = 3

      function funDoubleTankMM(x)
        f = zeros(eltype(x), 3,4)
        f[1,1] = -sqrt(x[1])
        f[2,1] = sqrt(x[1]) - sqrt(x[2])

        f[1,2] = 1

        f[1,3] = 0.5

        f[1,4] = 2
        return f
      end


    return name, t0, tf, x0, Q, funDoubleTankMM, lb, ub, n_omega

  elseif index==3
      name = "Egerstedt"
      t0 = 0.0
      tf = 1.0

      Q = [1.0 0.0;
           0.0 1.0]
      x0 = [0.5; 0.5]

      n_omega = 3

      function funEgerstedt(x)
        f = zeros(eltype(x), 2,4)

        f[1,2] = -x[1]
        f[2,2] = x[1] + 2*x[2]

        f[1,3] = x[1] + x[2]
        f[2,3] = x[1] - 2*x[2]

        f[1,4] = x[1] - x[2]
        f[2,4] = x[1] + x[2]
        return f
      end

      lb = -Inf*ones(length(x0))
      lb[1] = 0.4
      return name, t0, tf, x0, Q, funEgerstedt, lb, ub, n_omega

  elseif index == 4
      name = "LotkaVolterra"
      t0 = 0.0
      tf = 12.0

      C = [1.0 0.0 -1.0;
           0.0 1.0 -1.0]
      Q = C'*C
      x0 = [0.5; 0.7; 1.0]

      n_omega = 1

      function funLotkaVolterra(x)
        f = zeros(eltype(x), 3,2)
        f[1,1] = x[1] - x[1]*x[2]
        f[2,1] = -x[2] + x[1]*x[2]

        f[1,2] = - 0.4*x[1]
        f[2,2] = - 0.2*x[2]
        return f
      end

      return name, t0, tf, x0, Q,  funLotkaVolterra, lb, ub, n_omega

  elseif index == 5
      name = "Tank"
      t0 = 0.0
      tf = 10.0

      C = [0.0 1.0 -1.0]
      Q = C'*C
      x0 = [2.0; 2.0; 3.0]

      n_omega = 2
      function funTank(x)
        f = zeros(eltype(x), 3,3)

        f[1,1] = -sqrt(x[1])
        f[2,1] = sqrt(x[1]) - sqrt(x[2])
        f[3,1] = -0.05

        f[1,2] = 1.0

        f[1,3] = 2.0

        return f

      end

    return name, t0, tf, x0, Q, funTank, lb, ub, n_omega

  elseif index==6
      name = "ThreeTankMM"
      t0 = 0.0
      tf = 12.0

      C = [0.0 0.0 1.0 -1.0;
           0.0 1.0 0.0 -1.0;
           0.0 1.0 0.0 -1.0]
      Q = C'*C
      x0 = [2.0; 2.0; 2.0; 3.0]
      n_omega = 3

      function funThreeTankMM(x)
        f = zeros(eltype(x), 4,4)
        f[1,1] = -sqrt(x[1])
        f[2,1] = sqrt(x[1]) - sqrt(x[2])
        f[3,1] = sqrt(x[2]) - sqrt(x[3])

        f[1,2] = 1

        f[1,3] = 2

        f[1,4] = -sqrt(0.8*x[3])
        f[3,4] = sqrt(0.8*x[3])
        return f

      end
      return name, t0, tf, x0, Q, funThreeTankMM, lb, ub, n_omega


    elseif index==7
      # https://mintoc.de/index.php/Lotka_Volterra_Multimode_fishing_problem
      name = "LotkaVolterraMultiMode"
      t0 = 0.0
      tf = 12.0

      C = [0.0 1.0 -1.0;
           1.0 0.0 -1.0]
      Q = C'*C
      x0 = [0.5, 0.7, 1.0]

      n_omega = 3

      function funLotkaVolterraMultiMode(x)
        f = zeros(eltype(x), 3,4)
        f[1,1] = x[1] - x[1]*x[2]
        f[2,1] = -x[2] + x[1]*x[2]
        f[1,2] = - 0.2*x[1]
        f[2,2] = - 0.1*x[2]
        f[1,3] = - 0.4*x[1]
        f[2,3] = - 0.2*x[2]
        f[1,4] = - 0.01*x[1]
        f[2,4] = - 0.1*x[2]
        return f
      end

     return name, t0, tf, x0, Q, funLotkaVolterraMultiMode, lb, ub, n_omega

  else
      println("index is out of range")
  end
end
