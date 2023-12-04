using LinearAlgebra
using Plots, OrdinaryDiffEq, DiffEqCallbacks, RecursiveArrayTools, JLD2
cd(@__DIR__)

σ_RBF(x, c, σ) = exp( -norm(x - c)^2 / σ)
# Σ_RBF(x::Vector{Float64}, c_min, c_max, N_c, σ) = [σ_RBF(x, c, σ) for c in range(c_min, c_max, N_c) .* [ones(length(x))]]
Σ_RBF(x::Vector{Float64}, c_min, c_max, N_c, σ) = [σ_RBF(x, c, σ) for c in [[c_1,c_2] for c_1 in range(c_min, c_max, N_c) for c_2 in range(c_min, c_max, N_c)]]

function CL_dyn_out(X, p, t)
    # parameters
    A   = p.A
    B   = p.B
    b_Δ = p.b_Δ
    k_x = p.k_x
    k_r = p.k_r
    Γ_w = p.Γ_w
    κ   = p.κ
    τ_f = p.τ_f
    k_L = p.k_L
    k_U = p.k_U
    ϑ   = p.ϑ
    R   = p.R
    P   = p.P
    c_min = p.c_min
    c_max = p.c_max
    N_c   = p.N_c
    σ     = p.σ
    Ω_a   = p.Ω_a
    M_a   = p.M_a

   
    # signals
    (x, x_r, Ŵ, ξ, η, Ω, M) = X.x
    # (x, x_r, Ŵ) = X.x
    e = x_r - x

    # command
    # if t >= 12.0 && t <= 17.0
    #     ϕ_cmd = 1.0
    # elseif t >= 22.0 && t <= 27.0
    #     ϕ_cmd = -1.0
    # else
    #     ϕ_cmd = 0.0
    # end
    ϕ_cmd = sign(sin(0.5 * (t - 5.0)))

    # reference model
    A_r = A - B * k_x
    B_r = B * k_r
    dx_r_dt = A_r * x_r + B_r * ϕ_cmd

    # controller
    Φ_RBF = Σ_RBF(x, c_min, c_max, N_c, σ)
    u_ad = Ŵ' * Φ_RBF
    u = -k_x * x + k_r * ϕ_cmd - u_ad

    # plant dynamics
    Φ_Δ = [1.0, x[1], x[2], abs(x[1]) * x[2], abs(x[2]) * x[2], x[1]^3]
    Δ = b_Δ' * Φ_Δ
    dx_dt = A * x + B * (u + Δ)

    # adaptation law
    if rank(Ω_a) == N_c^2
        κ = 0.0
    end
    dŴ_dt = -Γ_w * (Φ_RBF * e' * P * B + R * (Ω_a * Ŵ - M_a) + κ * Ŵ)
    # dŴ_dt = -Γ_w * Φ_RBF * e' * P * B

    dξ_dt = ([u_ad + pinv(B) * (I(2) / τ_f + A_r) * e] - ξ) / τ_f
    dη_dt = (Φ_RBF - η) / τ_f

    k     = k_L + (k_U - k_L) * tanh(ϑ * norm(dη_dt))
    χ     = ξ - [pinv(B) * e] / τ_f
    dΩ_dt = -k * Ω + η * η'
    dM_dt = -k * M + η * χ'

    output = [ϕ_cmd, u, u_ad, norm(e), F_info(Ω_a), Δ]
    return dx_dt, dx_r_dt, dŴ_dt, dξ_dt, dη_dt, dΩ_dt, dM_dt, output
    # return dx_dt, dx_r_dt, dŴ_dt, output
end

function CL_dyn(dX_dt, X, p, t)
    (dx_dt, dx_r_dt, dŴ_dt, dξ_dt, dη_dt, dΩ_dt, dM_dt) = CL_dyn_out(X, p, t)[1:end-1]
    # (dx_dt, dx_r_dt, dŴ_dt) = CL_dyn_out(X, p, t)[1:end-1]

    dX_dt.x[1] .= dx_dt
    dX_dt.x[2] .= dx_r_dt
    dX_dt.x[3] .= dŴ_dt
    dX_dt.x[4] .= dξ_dt
    dX_dt.x[5] .= dη_dt
    dX_dt.x[6] .= dΩ_dt
    dX_dt.x[7] .= dM_dt
end

function F_info(X)
    λ = eigen(X).values
    λ[findall(λ .< 0.0)] .= 0.0
    return minimum(λ)
end

function exec_sim(; Γ_w = 100.0, τ_f = 0.01, N_c = 5, t_f = 40.0, R = 0.001)
    ϕ_0 = 0.0     # 6.0 * pi / 180.0
    ϕ_dot_0 = 0.0 # 419.4 * (pi / 180.0) / (4.0 * 15.0 / 0.429)

    p = (;  A   = [0.0 1.0;
                   0.0 0.0],
            B   = [0.0;
                   1.0],
            b_Δ = -[1.0, 0.2314, 0.6918, 0.6245, 0.1, 0.214],
            k_x = [4.0, 2.0]',
            k_r = 4.0,
            Γ_w = Γ_w,
            κ   = 0.001,
            τ_f = τ_f,
            k_L = 0.1,
            k_U = 1.0,
            ϑ   = 0.1,
            R   = R,
            c_min = -2.0,
            c_max =  2.0,
            N_c   = N_c,
            σ     =  5.0,
            Ω_a = zeros(N_c^2, N_c^2),
            M_a = zeros(N_c^2))
    A_r = p.A - p.B * p.k_x
    Q = I(2)
    p = merge(p, (; P = lyap(A_r', Q)))

    X_0 = ArrayPartition([ϕ_0, ϕ_dot_0], [ϕ_0, ϕ_dot_0], zeros(N_c^2), zeros(1), zeros(N_c^2), zeros(N_c^2, N_c^2), zeros(N_c^2))
    # X_0 = ArrayPartition([ϕ_0, ϕ_dot_0], [ϕ_0, ϕ_dot_0], zeros(N_c))

    prob = ODEProblem(CL_dyn, X_0, (0.0, t_f), p)
    function discr_condition(X, t, integrator)
        Ω   = X.x[6]
        Ω_a = integrator.p.Ω_a

        return F_info(Ω_a) <= F_info(Ω)
    end
    function affect!(integrator)
        integrator.p.Ω_a .= integrator.u.x[6]
        integrator.p.M_a .= integrator.u.x[7]
    end
    cb_F_info = DiscreteCallback(discr_condition, affect!)

    saved_values = SavedValues(Float64, Vector{Float64}) 
    cb_save = SavingCallback((X, t, integrator) -> CL_dyn_out(X, integrator.p, t)[end], saved_values, saveat = 0.01)
    cb = CallbackSet(cb_F_info, cb_save)

    sol = solve(prob, DP8(), callback = cb, reltol = 1e-8)

    return sol, saved_values.t, saved_values.saveval
end

function main(; case = 1)
    if case == 1
        Γ_w_list = [0.0, 1.0, 10.0, 100.0]
        R = 0.1
        test_case = Γ_w_list
    elseif case == 2
        Γ_w = 10.0
        R_list = [0.0, 0.01, 1.0, 100.0]
        test_case = R_list
    end
    res = Vector(undef,length(test_case))

    f_phi         = plot(xlabel="\$t\$", ylabel="\$\\phi\$ [rad]", legend = :best)
    f_phi_dot     = plot(xlabel="\$t\$", ylabel="\$\\dot{\\phi}\$ [rad/s]", legend = :best)
    f_u           = plot(xlabel = "\$t\$", ylabel = "\$u\$", legend = :best)
    f_norm_e      = plot(xlabel = "\$t\$", ylabel = "\$\\Vert e \\Vert\$", legend = :best)
    f_F_info      = plot(xlabel = "\$t\$", ylabel = "\$\\mathcal{F}\\left(\\Omega_{a}\\right)\$", legend = :best)
    f_tilde_Delta = plot(xlabel = "\$t\$", ylabel = "\$u_{ad} - \\Delta\$", legend = :best)
    f_Delta = plot(xlabel = "\$t\$", ylabel = "\$\\Delta\$", legend = :best)
    f_u_ad  = plot(xlabel = "\$t\$", ylabel = "\$u_{ad}\$", legend = :best)

    for i in eachindex(test_case)
        @show test_case[i]
        if case == 1
            sol, t_save, output = exec_sim(Γ_w = test_case[i], R = R, N_c = 5)
            label_string = "\$\\Gamma_{w} = $(round(Int, Γ_w_list[i]))\$"
        elseif case == 2
            sol, t_save, output = exec_sim(Γ_w = Γ_w, R = test_case[i], N_c = 5)
            label_string = "\$R = $(R_list[i])\$"
        end
        
        t     = sol.t
        ϕ     = [sol.u[i].x[1][1] for i in eachindex(sol)]
        ϕ_dot = [sol.u[i].x[1][2] for i in eachindex(sol)]
        
        output      = hcat(output...)'
        ϕ_cmd       = output[:,1]
        u           = output[:,2]
        u_ad        = output[:,3]
        norm_e      = output[:,4]
        F_info_Ω_a  = output[:,5]
        Δ           = output[:,6]
        
        plot!(f_phi, t, ϕ, label = label_string)
        if i == length(test_case)
            plot!(f_phi, t_save, ϕ_cmd, label = "cmd")
        end
        plot!(f_phi_dot, t, ϕ_dot, label = label_string)
        plot!(f_u, t_save, u, label = label_string)
        plot!(f_norm_e, t_save, norm_e, label = label_string)
        plot!(f_F_info, t_save, F_info_Ω_a, label = label_string)
        plot!(f_tilde_Delta, t_save, u_ad - Δ, label = label_string)
        plot!(f_Delta, t_save, Δ, label = label_string)
        plot!(f_u_ad, t_save, u_ad, label = label_string)

        res[i] = (sol, output)
    end
    f_xu = plot(f_phi, f_phi_dot, f_u, layout = (3,1), size = (600, 600))
    f_J  = plot(f_norm_e, f_tilde_Delta, f_F_info, layout = (3,1), size = (600, 600))
    jldsave("sim_data.jld2"; res)
    
    display(f_phi)
    display(f_phi_dot)
    display(f_u)
    display(f_norm_e)
    display(f_F_info)
    display(f_tilde_Delta)
    display(f_Delta)
    display(f_u_ad)
    display(f_xu)
    display(f_J)

    
    savefig(f_phi, "Fig_phi_C$(round(Int, case)).pdf")
    savefig(f_phi_dot, "Fig_phi_dot_C$(round(Int, case)).pdf")
    savefig(f_u, "Fig_u_C$(round(Int, case)).pdf")
    savefig(f_norm_e, "Fig_norm_e_C$(round(Int, case)).pdf")
    savefig(f_F_info, "Fig_F_info_C$(round(Int, case)).pdf")
    savefig(f_tilde_Delta, "Fig_tilde_Delta_C$(round(Int, case)).pdf")
    savefig(f_Delta, "Fig_Delta_C$(round(Int, case)).pdf")
    savefig(f_u_ad, "Fig_u_ad_C$(round(Int, case)).pdf")
    savefig(f_xu, "Fig_xu_C$(round(Int, case)).pdf")
    savefig(f_J, "Fig_J_C$(round(Int, case)).pdf")
end

##
main(case = 1)