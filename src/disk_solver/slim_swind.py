import math
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from parameter import set_const


class functions(object):
    """class includes all fuctions to solve slim disk"""

    def __init__(self, const):
        self.const = const
        self.vr_vs = []
        self.logdata = []
        # self.sigma=None
        # self.vr=None
        self.mdot_va = self.const.mdot

    def get_omega_k(self, R):
        return (self.const.G * self.const.M_bh / R) ** 0.5 / (R - self.const.Rs)

    def get_domega_dR(self, R):
        self.part1 = (self.const.G * self.const.M_bh / R) ** 0.5 / (
            R - self.const.Rs
        ) ** 2
        self.part2 = (
            self.const.G
            * self.const.M_bh
            / (
                2
                * R
                * R
                * (R - self.const.Rs)
                * (self.const.G * self.const.M_bh / R) ** 0.5
            )
        )
        return -(self.part1 + self.part2)

    def get_Mdot(self, R):
        rr = R / self.const.Rs
        Mdot_s = self.const.Mdot_0 * (rr / self.const.r_0) ** self.const.s
        return Mdot_s

    def get_dlnMdot_dr(self, R, Mdot):
        dlnM_dR = self.const.s / R
        return dlnM_dR

    def get_dlnomegak_dr(self, R):
        return -1 / 2 / R - 1 / (R - self.const.Rs)

    def get_w(self, R, l, lin, Mdot):
        return Mdot * (l - lin) / (2 * self.const.pi * self.const.alpha * R * R)

    def get_sigma(self, w, eta, R, Mdot):
        return Mdot**2 / (4 * self.const.pi**2 * R**2 * w * eta)

    def get_H(self, omega_k, w, sigma):
        # return 3*(w/sigma)**0.5/omega_k
        return (
            (2 * self.const.N + 2) * w * self.const.IN / sigma / self.const.IN1
        ) ** 0.5 / omega_k

    def get_P(self, w, H):
        return w / (2 * self.const.IN1 * H)

    def get_rho(self, sigma, H):
        return sigma / 2 / H / self.const.IN

    def b(self, rho):
        return self.const.Rg * rho / self.const.mu

    def functionT(self, t, b, c):
        return (self.const.a / 3) * t**4 + b * t + c

    def get_T(self, b, c):
        solve = fsolve(self.functionT, 1000, args=(b, c))
        if solve:
            for T in solve:
                if np.isreal(T) and T > 0:
                    return T
        else:
            print('get wrong when solve T')
            return 0

    def get_kappa(self, rho, T):
        kabs = 6.4e22 * (self.const.IN * rho) * (2 * T / 3) ** (-3.5)
        return self.const.kes + kabs

    def get_beta(self, rho, T, P):
        return self.const.Rg / self.const.mu * rho * T / P

    def get_gamma1(self, beta):
        return (32 - 24 * beta - 3 * beta**2) / (24 - 21 * beta)

    def get_gamma3(self, beta):
        return (32 - 27 * beta) / (24 - 21 * beta)

    def get_Fz(self, rho, H, kappa, T):
        return 4 * self.const.a * self.const.c * T**4 / (3 * kappa * rho * H)

    def get_K(self, sigma, w):
        return w / sigma

    def get_frad(self, rho, H, R):
        frad = self.const.G * self.const.M_bh * H * self.const.c / self.const.kes
        frad = frad / (R**2 + H**2) ** 0.5
        frad = frad / ((R**2 + H**2) ** 0.5 - self.const.Rs) ** 2
        return 2 * self.const.pi * R * frad

    def main_fuction(self, r, y, lin):
        R = r * self.const.Rs
        l, eta = y[0], abs(y[1])
        Mdot = self.get_Mdot(R)
        dlnMdot_dr = self.get_dlnMdot_dr(R, Mdot)
        omega_k = self.get_omega_k(R)
        dlnomegak_dr = self.get_dlnomegak_dr(R)
        w = self.get_w(R, l, lin, Mdot)
        sigma = self.get_sigma(w, eta, R, Mdot)
        H = self.get_H(omega_k, w, sigma)
        P = self.get_P(w, H)
        rho = self.get_rho(sigma, H)
        T = self.get_T(b=self.b(rho), c=-P)
        kappa = self.get_kappa(rho, T)
        beta = self.get_beta(rho, T, P)
        gamma1 = self.get_gamma1(beta)
        gamma3 = self.get_gamma3(beta)
        Fz = self.get_Fz(rho, H, kappa, T)
        K = self.get_K(sigma, w)
        cs = (P / rho) ** 0.5
        Teff = (Fz / self.const.sb) ** 0.25
        vr = -Mdot / 2 / self.const.pi / R / sigma
        vr_c = vr / self.const.c
        vs = cs / self.const.c
        lk = omega_k * R**2
        Qrad = 4 * self.const.pi * R * Fz
        # frad = self.get_frad(rho, H, R)
        # Qrad = min(Qrad, 2*frad)
        fun1 = (
            (l**2 - lk**2) / (K * R**3)
            - dlnomegak_dr
            + 2 * (1 + eta) / R
            - eta / R
            - dlnMdot_dr
        )
        u1 = (gamma1 + 1) / ((gamma3 - 1) * R)
        u2 = (gamma1 - 1) * dlnMdot_dr / (gamma3 - 1)
        u3 = (gamma1 - 1) * dlnomegak_dr / (gamma3 - 1)
        u4 = Qrad / (K * Mdot)
        u5 = 4 * self.const.pi * self.const.alpha * w * l / (K * Mdot * R)
        u6 = (3 * gamma1 - 1) * fun1 / (2 * eta * (gamma3 - 1))
        part1 = u1 + u2 + u3 + u4 - u5 - u6
        part2 = Mdot * gamma1 / (
            R * R * self.const.pi * self.const.alpha * w * (gamma3 - 1)
        ) - 2 * self.const.pi * self.const.alpha * w / (Mdot * K)
        part2 = part2 - (1 + eta) * Mdot * (3 * gamma1 - 1) / (
            2 * self.const.pi * self.const.alpha * w * R * R * 2 * eta * (gamma3 - 1)
        )
        dl_dR = part1 / part2
        fun2 = -(1 + eta) * Mdot / (2 * self.const.pi * self.const.alpha * w * R * R)
        deta_dR = fun1 + fun2 * dl_dR
        dl_dr = dl_dR * self.const.Rs
        deta_dr = deta_dR * self.const.Rs
        self.vr_vs.append(abs(vr_c / vs))

        self.logdata.append(
            (
                r,
                R,
                l,
                eta,
                dl_dR,
                deta_dR,
                omega_k,
                w,
                sigma,
                H,
                beta,
                Teff,
                T,
                vr,
                vs,
                Fz,
                gamma1,
                gamma3,
            )
        )
        return np.array([dl_dr, deta_dr])

    def intial_value(self, r_0, lin):
        R_0 = r_0 * self.const.Rs
        l_0 = (self.const.G * self.const.M_bh * R_0) ** 0.5 / (
            1 - self.const.Rs / R_0
        ) ** (-1)
        vr = (
            -(5.4e5)
            * self.const.alpha**0.8
            * self.const.mdot**0.3
            * r_0 ** (-0.25)
            * self.const.m ** (-0.2)
        )
        sigma = -self.const.Mdot_0 / (2 * self.const.pi * vr * R_0)
        w_0 = (
            self.const.Mdot_0
            * (l_0 - lin)
            / 2
            / self.const.pi
            / self.const.alpha
            / R_0**2
        )
        eta_0 = vr**2 * sigma / w_0

        return np.array([l_0, eta_0])

    def lin(self, lint):
        return lint * self.const.Rs * self.const.c


def solve(const):
    lintmin = 1
    lintmax = 2
    lint = const.lint
    fuc = functions(const)
    t_array = np.arange(const.r_0, const.r_end, -0.01)
    i = 0
    while True:
        fuc.vr_vs.clear()
        fuc.logdata.clear()
        lin = fuc.lin(lint)
        y0 = fuc.intial_value(const.r_0, lin)
        solve, solveinfo = odeint(
            func=fuc.main_fuction,
            y0=y0,
            t=t_array,
            args=(lin,),
            full_output=True,
            printmessg=True,
            tfirst=True,
            atol=1e-10,
            rtol=1e-10,
        )
        tcur = solveinfo['tcur']
        t_solve_index = np.nonzero(tcur)
        min_solve_t = np.min(tcur[t_solve_index])

        if min_solve_t > 3:
            lintmax = lint
        else:
            if max(fuc.vr_vs) < 1:
                lintmin = lint
            else:
                print('get solve! lint:', lint)

                break
        lint = (lintmax + lintmin) / 2
        i = i + 1
        print('{} -time Trail,next lint={}'.format(i, lint))
    t_array = t_array[t_solve_index]
    solve_array = solve[t_solve_index]
    logdata = np.array(
        fuc.logdata,
        dtype={
            'names': [
                'r',
                'R',
                'l',
                'eta',
                'dl_dR',
                'deta_dR',
                'omega_k',
                'w',
                'sigma',
                'H',
                'beta',
                'Teff',
                'T',
                'vr',
                'vs',
                'Fz',
                'gamma1',
                'gamma3',
            ],
            'formats': ['f8'] * 18,
        },
    )
    return logdata
