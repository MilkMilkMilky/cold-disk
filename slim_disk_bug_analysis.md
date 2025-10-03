# Slim Disk 求解器 Bug 分析与修复

## 问题描述

1. **旧求解器行为**: 当 `l_in > 1.8` 时，`vr/cs` 可以跨越 1，打靶法成功收敛
2. **新求解器问题**: 无论 `l_in` 多大，`vr/cs` 最高只能达到 0.9，打靶法失败
3. **可能原因**: 新算法中描述方程相关的代码存在问题

## 方法检查列表

基于之前的方法映射表，我们需要逐个检查每个方法从旧实现到新实现的迁移是否正确。

### 4.1 基本物理量计算方法

| 序号 | 旧实现方法                   | 新实现方法                                                      | 检查状态  | 问题描述         |
| ---- | ---------------------------- | --------------------------------------------------------------- | --------- | ---------------- |
| 1    | `get_omega_k(R)`             | `get_slim_angvelk(par, radius)`                                 | 🔍 待检查 |                  |
| 2    | `get_domega_dR(R)`           | ❌ (未直接实现)                                                 | 🔍 待检查 | 可能影响微分方程 |
| 3    | `get_Mdot(R)`                | `get_slim_accrate(par, radius)`                                 | 🔍 待检查 |                  |
| 4    | `get_dlnMdot_dr(R, Mdot)`    | `get_slim_dlnaccrate_dradius(par, radius)`                      | 🔍 待检查 |                  |
| 5    | `get_dlnomegak_dr(R)`        | `get_slim_dlnangvelk_dradius(par, radius)`                      | 🔍 待检查 |                  |
| 6    | `get_w(R, l, lin, Mdot)`     | `get_slim_arealpressure(par, radius, angmom, angmomin)`         | 🔍 待检查 |                  |
| 7    | `get_sigma(w, eta, R, Mdot)` | `get_slim_arealdensity(par, arealpressure, coff_eta, radius)`   | 🔍 待检查 |                  |
| 8    | `get_H(omega_k, w, sigma)`   | `get_slim_halfheight(par, radius, arealpressure, arealdensity)` | 🔍 待检查 |                  |
| 9    | `get_P(w, H)`                | `get_slim_pressure(par, arealpressure, halfheight)`             | 🔍 待检查 |                  |
| 10   | `get_rho(sigma, H)`          | `get_slim_density(par, arealdensity, halfheight)`               | 🔍 待检查 |                  |

### 4.2 热力学和辐射方法

| 序号 | 旧实现方法                 | 新实现方法                                                  | 检查状态  | 问题描述 |
| ---- | -------------------------- | ----------------------------------------------------------- | --------- | -------- |
| 11   | `get_T(b, c)`              | `get_slim_temperature(pressure, density)`                   | 🔍 待检查 |          |
| 12   | `get_kappa(rho, T)`        | `get_slim_opacity(par, density, temperature)`               | 🔍 待检查 |          |
| 13   | `get_beta(rho, T, P)`      | `get_slim_pressure_ratio(density, temperature, pressure)`   | 🔍 待检查 |          |
| 14   | `get_gamma1(beta)`         | `get_slim_chandindex_1(pressure_ratio)`                     | 🔍 待检查 |          |
| 15   | `get_gamma3(beta)`         | `get_slim_chandindex_3(pressure_ratio)`                     | 🔍 待检查 |          |
| 16   | `get_Fz(rho, H, kappa, T)` | `get_slim_fluxz(density, halfheight, opacity, temperature)` | 🔍 待检查 |          |
| 17   | `get_K(sigma, w)`          | `get_slim_apresstoadens(arealpressure, arealdensity)`       | 🔍 待检查 |          |

### 4.3 速度和运动学方法

| 序号 | 旧实现方法    | 新实现方法                                   | 检查状态  | 问题描述 |
| ---- | ------------- | -------------------------------------------- | --------- | -------- |
| 18   | ❌ (内联计算) | `get_slim_radvel(par, radius, arealdensity)` | 🔍 待检查 |          |
| 19   | ❌ (内联计算) | `get_slim_soundvel(pressure, density)`       | 🔍 待检查 |          |
| 20   | ❌ (内联计算) | `get_slim_rveltosvel(radvel, soundvel)`      | 🔍 待检查 |          |

### 4.4 求解器核心方法

| 序号 | 旧实现方法                | 新实现方法                                           | 检查状态  | 问题描述         |
| ---- | ------------------------- | ---------------------------------------------------- | --------- | ---------------- |
| 21   | `main_fuction(r, y, lin)` | `slim_disk_model(indep_var, dep_var, par, angmomin)` | 🔍 待检查 | **核心微分方程** |
| 22   | `intial_value(r_0, lin)`  | `get_slim_initvalue(par)`                            | 🔍 待检查 |                  |
| 23   | `lin(lint)`               | `get_slim_angmomin(par, dimless_angmomin)`           | 🔍 待检查 |                  |

## 详细检查结果

### 检查 1: `get_omega_k(R)` vs `get_slim_angvelk(par, radius)`

#### 旧实现

```python
def get_omega_k(self, R):
    return (self.const.G * self.const.M_bh / R) ** 0.5 / (R - self.const.Rs)
```

#### 新实现

```python
@staticmethod
def get_slim_angvelk(*, par: DiskParams, radius) -> float | np.ndarray:
    radius = np.asarray(radius)
    bhmass = DiskTools.get_bhmass(par=par)
    radius_sch = DiskTools.get_radius_sch(par=par)
    slim_angvelk = np.sqrt(cgs_consts.cgs_gra * bhmass / radius) / (radius - radius_sch)
    return slim_angvelk
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致

---

### 检查 2: `get_domega_dR(R)` vs ❌ (未直接实现)

#### 旧实现

```python
def get_domega_dR(self, R):
    self.part1 = (self.const.G * self.const.M_bh / R) ** 0.5 / (R - self.const.Rs) ** 2
    self.part2 = (
        self.const.G * self.const.M_bh
        / (2 * R * R * (R - self.const.Rs) * (self.const.G * self.const.M_bh / R) ** 0.5)
    )
    return -(self.part1 + self.part2)
```

#### 新实现

❌ **未直接实现**

**检查状态**: ✅ **不是问题**
**说明**: 该方法在旧求解器中完全没有被使用，因此在新求解器中移除是正确的

---

### 检查 3: `get_Mdot(R)` vs `get_slim_accrate(par, radius)`

#### 旧实现

```python
def get_Mdot(self, R):
    rr = R / self.const.Rs
    Mdot_s = self.const.Mdot_0 * (rr / self.const.r_0) ** self.const.s
    return Mdot_s
```

#### 新实现

```python
@staticmethod
def get_slim_accrate(*, par: DiskParams, radius) -> float | np.ndarray:
    radius = np.asarray(radius)
    radius_sch = DiskTools.get_radius_sch(par=par)
    dimless_radius = radius / radius_sch
    accrate_out = DiskTools.get_accrate_fromdimless(par=par, dimless_accrate=par.dimless_accrate)
    slim_accrate = accrate_out * np.float_power((dimless_radius / par.dimless_radius_out), par.wind_index)
    return slim_accrate
```

**检查状态**: ✅ **基本正确**
**潜在问题**:

1. 公式结构相同: `Mdot_0 * (r/r_0)^s`
2. 需要确认 `DiskTools.get_accrate_fromdimless()` 的实现是否正确

---

### 检查 4: `get_dlnMdot_dr(R, Mdot)` vs `get_slim_dlnaccrate_dradius(par, radius)`

#### 旧实现

```python
def get_dlnMdot_dr(self, R, Mdot):
    dlnM_dR = self.const.s / R
    return dlnM_dR
```

#### 新实现

```python
@staticmethod
def get_slim_dlnaccrate_dradius(*, par: DiskParams, radius) -> float | np.ndarray:
    radius = np.asarray(radius)
    slim_dlnaccrate_dradius = par.wind_index / radius
    return slim_dlnaccrate_dradius
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致 (`s/R` vs `wind_index/radius`)

---

### 检查 5: `get_dlnomegak_dr(R)` vs `get_slim_dlnangvelk_dradius(par, radius)`

#### 旧实现

```python
def get_dlnomegak_dr(self, R):
    return -1 / 2 / R - 1 / (R - self.const.Rs)
```

#### 新实现

```python
@staticmethod
def get_slim_dlnangvelk_dradius(*, par: DiskParams, radius) -> float | np.ndarray:
    radius = np.asarray(radius)
    radius_sch = DiskTools.get_radius_sch(par=par)
    slim_dlnangvelk_dradius = -1 / 2 / radius - 1 / (radius - radius_sch)
    return slim_dlnangvelk_dradius
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致

---

## 初步检查总结 (前 5 个方法)

### 发现的问题:

1. **缺失方法**: `get_domega_dR(R)` 在新实现中没有直接对应方法
2. **依赖方法**: 需要验证 `DiskTools.get_accrate_fromdimless()` 的正确性

### 下一步检查重点:

1. 继续检查剩余的基本物理量计算方法
2. 重点关注微分方程相关的计算
3. 检查是否有其他缺失的导数计算

---

### 检查 6: `get_w(R, l, lin, Mdot)` vs `get_slim_arealpressure(par, radius, angmom, angmomin)`

#### 旧实现

```python
def get_w(self, R, l, lin, Mdot):
    return Mdot * (l - lin) / (2 * self.const.pi * self.const.alpha * R * R)
```

#### 新实现

```python
@staticmethod
def get_slim_arealpressure(*, par: DiskParams, radius, angmom, angmomin) -> float | np.ndarray:
    radius, angmom = np.asarray(radius), np.asarray(angmom)
    slim_accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
    slim_arealpressure = slim_accrate * (angmom - angmomin) / (2 * math.pi * par.alpha_viscosity * radius * radius)
    return slim_arealpressure
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`Mdot * (l - lin) / (2π * α * R²)`

---

### 检查 7: `get_sigma(w, eta, R, Mdot)` vs `get_slim_arealdensity(par, arealpressure, coff_eta, radius)`

#### 旧实现

```python
def get_sigma(self, w, eta, R, Mdot):
    return Mdot**2 / (4 * self.const.pi**2 * R**2 * w * eta)
```

#### 新实现

```python
@staticmethod
def get_slim_arealdensity(*, par: DiskParams, arealpressure, coff_eta, radius) -> float | np.ndarray:
    arealpressure = np.asarray(arealpressure)
    coff_eta, radius = np.asarray(coff_eta), np.asarray(radius)
    slim_accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
    slim_arealdensity = slim_accrate * slim_accrate / (4 * math.pi**2 * radius * radius * arealpressure * coff_eta)
    return slim_arealdensity
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`Mdot² / (4π² * R² * w * η)`

---

### 检查 8: `get_H(omega_k, w, sigma)` vs `get_slim_halfheight(par, radius, arealpressure, arealdensity)`

#### 旧实现

```python
def get_H(self, omega_k, w, sigma):
    # return 3*(w/sigma)**0.5/omega_k
    return (
        (2 * self.const.N + 2) * w * self.const.IN / sigma / self.const.IN1
    ) ** 0.5 / omega_k
```

#### 新实现

```python
@staticmethod
def get_slim_halfheight(*, par: DiskParams, radius, arealpressure, arealdensity) -> float | np.ndarray:
    radius = np.asarray(radius)
    arealpressure, arealdensity = np.asarray(arealpressure), np.asarray(arealdensity)
    angvelk = SlimDisk.get_slim_angvelk(par=par, radius=radius)
    coff_k = np.sqrt(arealpressure / arealdensity)
    slim_halfheight = np.sqrt(2 * par.gas_index + 3) * coff_k / angvelk
    return slim_halfheight
```

**检查状态**: ⚠️ **潜在问题** (待单元测试验证)
**潜在问题**:

1. 旧实现: `√[(2N+2) * w * IN / (σ * IN1)] / Ωₖ`
2. 新实现: `√(2N+3) * √(w/σ) / Ωₖ`
3. **系数差异**: `2N+2` vs `2N+3` (可能是代数化简)
4. **积分系数处理**: 旧实现有 `IN/IN1`，新实现没有 (可能是代数化简)

---

### 检查 9: `get_P(w, H)` vs `get_slim_pressure(par, arealpressure, halfheight)`

#### 旧实现

```python
def get_P(self, w, H):
    return w / (2 * self.const.IN1 * H)
```

#### 新实现

```python
@staticmethod
def get_slim_pressure(*, par: DiskParams, arealpressure, halfheight) -> float | np.ndarray:
    arealpressure, halfheight = np.asarray(arealpressure), np.asarray(halfheight)
    coff_in1 = DiskTools.get_coeff_in(index=par.gas_index + 1)
    slim_pressure = arealpressure / 2 / halfheight / coff_in1
    return slim_pressure
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`w / (2 * IN1 * H)`

---

### 检查 10: `get_rho(sigma, H)` vs `get_slim_density(par, arealdensity, halfheight)`

#### 旧实现

```python
def get_rho(sigma, H):
    return sigma / 2 / H / self.const.IN
```

#### 新实现

```python
@staticmethod
def get_slim_density(*, par: DiskParams, arealdensity, halfheight) -> float | np.ndarray:
    arealdensity, halfheight = np.asarray(arealdensity), np.asarray(halfheight)
    coff_in = DiskTools.get_coeff_in(index=par.gas_index)
    slim_density = arealdensity / 2 / halfheight / coff_in
    return slim_density
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`σ / (2 * H * IN)`

---

## 检查总结 (方法 6-10)

### 发现的重要问题:

1. **检查 8 - 半高度计算**: `get_H` vs `get_slim_halfheight` 存在**显著差异**
   - 系数不一致: `2N+2` vs `2N+3`
   - 积分系数处理不同: 旧实现有 `IN/IN1`，新实现没有

### 无问题的方法:

- 检查 6: 面压力计算 ✅
- 检查 7: 面密度计算 ✅
- 检查 9: 压力计算 ✅
- 检查 10: 密度计算 ✅

**这个半高度计算的差异可能是导致 vr/cs 无法超过 0.9 的关键问题！**

---

### 检查 11: `get_T(b, c)` vs `get_slim_temperature(pressure, density)`

#### 旧实现

```python
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
```

#### 新实现

```python
@staticmethod
def get_slim_temperature(*, pressure: float, density: float) -> float:
    coff_b = cgs_consts.cgs_rg * density / cgs_consts.cgs_amm
    coff_c = -pressure

    def slim_temperature_func(t):
        return (cgs_consts.cgs_a / 3) * t**4 + coff_b * t + coff_c

    try:
        temperature = sp.optimize.root_scalar(
            slim_temperature_func,
            bracket=[1e-10, 1e8],
            method="brentq",
        )
        if temperature.converged and temperature.root > 0:
            slim_temperature = temperature.root
            return slim_temperature
    except Exception as e:
        warnings.warn(f"root_scalar failed with exception: {e}. Falling back to fsolve.", stacklevel=2)

    temperature_array = sp.optimize.fsolve(slim_temperature_func, 10000)
    for temp in temperature_array:
        if temp > 0:
            slim_temperature = temp
            return slim_temperature
    raise RuntimeError(f"Failed to solve slim temperature. pressure={pressure}, density={density}")
```

**检查状态**: ✅ **基本正确**
**潜在问题**:

1. 求解方法升级：`fsolve` → `root_scalar` (带回退)
2. 参数计算方式相同：`b = Rg*ρ/μ`, `c = -P`
3. 方程形式一致：`(a/3)*T⁴ + b*T + c = 0`

---

### 检查 12: `get_kappa(rho, T)` vs `get_slim_opacity(par, density, temperature)`

#### 旧实现

```python
def get_kappa(self, rho, T):
    kabs = 6.4e22 * (self.const.IN * rho) * (2 * T / 3) ** (-3.5)
    return self.const.kes + kabs
```

#### 新实现

```python
@staticmethod
def get_slim_opacity(*, par: DiskParams, density, temperature) -> float | np.ndarray:
    density, temperature = np.asarray(density), np.asarray(temperature)
    coff_in = DiskTools.get_coeff_in(index=par.gas_index)
    opacity_abs = cgs_consts.cgs_kra * (coff_in * density)
    opacity_abs *= np.float_power((2 * temperature / 3), (-3.5))
    slim_opacity = cgs_consts.cgs_kes + opacity_abs
    return slim_opacity
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`kes + kra * (IN * ρ) * (2T/3)^(-3.5)`

---

### 检查 13: `get_beta(rho, T, P)` vs `get_slim_pressure_ratio(density, temperature, pressure)`

#### 旧实现

```python
def get_beta(self, rho, T, P):
    return self.const.Rg / self.const.mu * rho * T / P
```

#### 新实现

```python
@staticmethod
def get_slim_pressure_ratio(*, density, temperature, pressure) -> float | np.ndarray:
    pressure, density = np.asarray(pressure), np.asarray(density)
    temperature = np.asarray(temperature)
    slim_pressure_ratio = cgs_consts.cgs_rg / cgs_consts.cgs_amm * density * temperature / pressure
    return slim_pressure_ratio
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`(Rg/μ) * ρ * T / P`

---

### 检查 14: `get_gamma1(beta)` vs `get_slim_chandindex_1(pressure_ratio)`

#### 旧实现

```python
def get_gamma1(self, beta):
    return (32 - 24 * beta - 3 * beta**2) / (24 - 21 * beta)
```

#### 新实现

```python
@staticmethod
def get_slim_chandindex_1(*, pressure_ratio) -> float | np.ndarray:
    pressure_ratio = np.asarray(pressure_ratio)
    slim_chandindex_1 = (32 - 24 * pressure_ratio - 3 * pressure_ratio**2) / (24 - 21 * pressure_ratio)
    return slim_chandindex_1
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式完全一致

---

### 检查 15: `get_gamma3(beta)` vs `get_slim_chandindex_3(pressure_ratio)`

#### 旧实现

```python
def get_gamma3(self, beta):
    return (32 - 27 * beta) / (24 - 21 * beta)
```

#### 新实现

```python
@staticmethod
def get_slim_chandindex_3(*, pressure_ratio) -> float | np.ndarray:
    pressure_ratio = np.asarray(pressure_ratio)
    slim_chandindex_3 = (32 - 27 * pressure_ratio) / (24 - 21 * pressure_ratio)
    return slim_chandindex_3
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式完全一致

---

## 检查总结 (方法 11-15)

### 无问题的方法:

- ✅ 检查 11: 温度求解 (求解器升级但逻辑一致)
- ✅ 检查 12: 不透明度计算
- ✅ 检查 13: 压力比计算
- ✅ 检查 14: 钱德拉塞卡指数 1
- ✅ 检查 15: 钱德拉塞卡指数 3

### 发现的问题:

**无新问题发现**，这批热力学和辐射方法的迁移都是正确的。

---

### 检查 16: `get_Fz(rho, H, kappa, T)` vs `get_slim_fluxz(density, halfheight, opacity, temperature)`

#### 旧实现

```python
def get_Fz(self, rho, H, kappa, T):
    return 4 * self.const.a * self.const.c * T**4 / (3 * kappa * rho * H)
```

#### 新实现

```python
@staticmethod
def get_slim_fluxz(*, density, halfheight, opacity, temperature) -> float | np.ndarray:
    density, halfheight = np.asarray(density), np.asarray(halfheight)
    opacity, temperature = np.asarray(opacity), np.asarray(temperature)
    slim_fluxz = 4 * cgs_consts.cgs_a * cgs_consts.cgs_c * np.float_power(temperature, 4)
    slim_fluxz /= 3 * opacity * density * halfheight
    return slim_fluxz
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`4 * a * c * T⁴ / (3 * κ * ρ * H)`

---

### 检查 17: `get_K(sigma, w)` vs `get_slim_apresstoadens(arealpressure, arealdensity)`

#### 旧实现

```python
def get_K(self, sigma, w):
    return w / sigma
```

#### 新实现

```python
@staticmethod
def get_slim_apresstoadens(arealpressure, arealdensity) -> float | np.ndarray:
    arealpressure, arealdensity = np.asarray(arealpressure), np.asarray(arealdensity)
    slim_apresstoadens = arealpressure / arealdensity
    return slim_apresstoadens
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`w / σ`

---

### 检查 18: ❌ (内联计算) vs `get_slim_radvel(par, radius, arealdensity)`

#### 旧实现 (在 main_function 中内联)

```python
vr = -Mdot / 2 / self.const.pi / R / sigma
```

#### 新实现

```python
@staticmethod
def get_slim_radvel(*, par: DiskParams, radius, arealdensity) -> float | np.ndarray:
    radius, arealdensity = np.asarray(radius), np.asarray(arealdensity)
    accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
    slim_radvel = -accrate / 2 / math.pi / radius / arealdensity
    return slim_radvel
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`-Mdot / (2π * R * σ)`

---

### 检查 19: ❌ (内联计算) vs `get_slim_soundvel(pressure, density)`

#### 旧实现 (在 main_function 中内联)

```python
cs = (P / rho) ** 0.5
```

#### 新实现

```python
@staticmethod
def get_slim_soundvel(*, pressure, density) -> float | np.ndarray:
    pressure, density = np.asarray(pressure), np.asarray(density)
    slim_soundvel = np.sqrt(pressure / density)
    return slim_soundvel
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`√(P / ρ)`

---

### 检查 20: ❌ (内联计算) vs `get_slim_rveltosvel(radvel, soundvel)`

#### 旧实现 (在 main_function 中内联)

```python
vr_c = vr / self.const.c
vs = cs / self.const.c
# 然后存储 abs(vr_c / vs) 到 vr_vs 列表
self.vr_vs.append(abs(vr_c / vs))
```

#### 新实现

```python
@staticmethod
def get_slim_rveltosvel(*, radvel, soundvel) -> float | np.ndarray:
    radvel, soundvel = np.asarray(radvel), np.asarray(soundvel)
    slim_rveltosvel = np.abs(radvel / soundvel)
    return slim_rveltosvel
```

**检查状态**: ⚠️ **潜在问题**
**潜在问题**:

1. 旧实现: `abs(vr/c) / (cs/c) = abs(vr/cs)`
2. 新实现: `abs(vr/cs)`
3. **计算逻辑一致**，但旧实现中有额外的光速归一化步骤
4. **可能不是问题**，因为 `c` 在分子分母中约掉了

---

## 检查总结 (方法 16-20)

### 无问题的方法:

- ✅ 检查 16: 垂直通量计算
- ✅ 检查 17: 面压力密度比计算
- ✅ 检查 18: 径向速度计算
- ✅ 检查 19: 声速计算

### 潜在问题:

- ⚠️ 检查 20: 速度比计算 (逻辑一致但实现细节略有不同)

**现在需要检查最关键的求解器核心方法 (21-23)！这些可能是问题的真正来源。**

---

### 检查 21: `main_fuction(r, y, lin)` vs `slim_disk_model(indep_var, dep_var, par, angmomin)` 🔥**核心微分方程**

这是最关键的检查！让我详细对比微分方程的实现。

#### 旧实现 (main_fuction)

```python
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

    # 微分方程组
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

    return np.array([dl_dr, deta_dr])
```

#### 新实现 (slim_disk_model)

```python
@staticmethod
def slim_disk_model(indep_var: float, dep_var: np.ndarray, par: DiskParams, angmomin: float) -> np.ndarray:
    radius_sch = DiskTools.get_radius_sch(par=par)
    radius = indep_var * radius_sch
    angmom, coff_eta = dep_var[0], abs(dep_var[1])
    accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
    dlnaccrate_dradius = SlimDisk.get_slim_dlnaccrate_dradius(par=par, radius=radius)
    dlnangvelk_dradius = SlimDisk.get_slim_dlnangvelk_dradius(par=par, radius=radius)
    arealpressure = SlimDisk.get_slim_arealpressure(par=par, radius=radius, angmom=angmom, angmomin=angmomin)
    arealdensity = SlimDisk.get_slim_arealdensity(
        par=par, arealpressure=arealpressure, coff_eta=coff_eta, radius=radius,
    )
    halfheight = SlimDisk.get_slim_halfheight(
        par=par, radius=radius, arealpressure=arealpressure, arealdensity=arealdensity,
    )
    pressure = SlimDisk.get_slim_pressure(par=par, arealpressure=arealpressure, halfheight=halfheight)
    density = SlimDisk.get_slim_density(par=par, arealdensity=arealdensity, halfheight=halfheight)
    temperature = SlimDisk.get_slim_temperature(pressure=float(pressure), density=float(density))
    opacity = SlimDisk.get_slim_opacity(par=par, density=density, temperature=temperature)
    pressure_ratio = SlimDisk.get_slim_pressure_ratio(density=density, temperature=temperature, pressure=pressure)
    chandindex_1 = SlimDisk.get_slim_chandindex_1(pressure_ratio=pressure_ratio)
    chandindex_3 = SlimDisk.get_slim_chandindex_3(pressure_ratio=pressure_ratio)
    fluxz = SlimDisk.get_slim_fluxz(density=density, halfheight=halfheight, opacity=opacity, temperature=temperature,)
    apresstoadens = SlimDisk.get_slim_apresstoadens(arealpressure=arealpressure, arealdensity=arealdensity)
    angmomk = SlimDisk.get_slim_angmomk(par=par, radius=radius)
    energy_qrad = 4 * math.pi * radius * fluxz

    # 微分方程组
    fun1 = (
        (angmom**2 - angmomk**2) / (apresstoadens * radius**3)
        - dlnangvelk_dradius
        + 2 * (1 + coff_eta) / radius
        - coff_eta / radius
        - dlnaccrate_dradius
    )
    u1 = (chandindex_1 + 1) / ((chandindex_3 - 1) * radius)
    u2 = (chandindex_1 - 1) * dlnaccrate_dradius / (chandindex_3 - 1)
    u3 = (chandindex_1 - 1) * dlnangvelk_dradius / (chandindex_3 - 1)
    u4 = energy_qrad / apresstoadens / accrate
    u5 = 4 * math.pi * par.alpha_viscosity * arealpressure * angmom / (apresstoadens * accrate * radius)
    u6 = (3 * chandindex_1 - 1) * fun1 / (2 * coff_eta * (chandindex_3 - 1))
    part1 = u1 + u2 + u3 + u4 - u5 - u6
    d1 = (
        accrate * chandindex_1
        / (radius * radius * math.pi * par.alpha_viscosity * arealpressure * (chandindex_3 - 1))
    )
    d2 = 2 * math.pi * par.alpha_viscosity * arealpressure / (accrate * apresstoadens)
    d3 = (
        (1 + coff_eta) * accrate * (3 * chandindex_1 - 1)
        / (2 * math.pi * par.alpha_viscosity * arealpressure * radius * radius * 2 * coff_eta * (chandindex_3 - 1))
    )
    part2 = d1 - d2 - d3
    dangmom_dradius = part1 / part2
    fun2 = -(1 + coff_eta) * accrate / (2 * math.pi * par.alpha_viscosity * arealpressure * radius * radius)
    dcoffeta_dradius = fun1 + fun2 * dangmom_dradius
    dangmom_ddimlessradius = dangmom_dradius * radius_sch
    dcoffeta_ddimlessradius = dcoffeta_dradius * radius_sch
    deri_arr = np.array([dangmom_ddimlessradius, dcoffeta_ddimlessradius])
    return deri_arr
```

**检查状态**: ✅ **微分方程计算无问题**

### **详细检查 part2 计算结构**

#### 旧实现的 part2:

```python
part2 = Mdot * gamma1 / (R * R * self.const.pi * self.const.alpha * w * (gamma3 - 1))  # term1
part2 = part2 - 2 * self.const.pi * self.const.alpha * w / (Mdot * K)                  # - term2
part2 = part2 - (1 + eta) * Mdot * (3 * gamma1 - 1) / (
    2 * self.const.pi * self.const.alpha * w * R * R * 2 * eta * (gamma3 - 1)
)  # - term3
```

分解为三项:

- **term1**: `Mdot * γ₁ / (R² * π * α * w * (γ₃ - 1))`
- **term2**: `2 * π * α * w / (Mdot * K)`
- **term3**: `(1 + η) * Mdot * (3γ₁ - 1) / (2 * π * α * w * R² * 2η * (γ₃ - 1))`

#### 新实现的 part2:

```python
d1 = accrate * chandindex_1 / (radius * radius * math.pi * par.alpha_viscosity * arealpressure * (chandindex_3 - 1))
d2 = 2 * math.pi * par.alpha_viscosity * arealpressure / (accrate * apresstoadens)
d3 = (1 + coff_eta) * accrate * (3 * chandindex_1 - 1) / (
    2 * math.pi * par.alpha_viscosity * arealpressure * radius * radius * 2 * coff_eta * (chandindex_3 - 1)
)
part2 = d1 - d2 - d3
```

分解为三项:

- **d1**: `accrate * chandindex_1 / (radius² * π * α * arealpressure * (chandindex_3 - 1))`
- **d2**: `2 * π * α * arealpressure / (accrate * apresstoadens)`
- **d3**: `(1 + coff_eta) * accrate * (3 * chandindex_1 - 1) / (2 * π * α * arealpressure * radius² * 2 * coff_eta * (chandindex_3 - 1))`

### **逐项对比分析**:

#### **第一项对比** (term1 vs d1):

- 旧: `Mdot * γ₁ / (R² * π * α * w * (γ₃ - 1))`
- 新: `accrate * chandindex_1 / (radius² * π * α * arealpressure * (chandindex_3 - 1))`
- **结论**: ✅ **完全等价** (变量名不同但含义相同)

#### **第二项对比** (term2 vs d2):

- 旧: `2 * π * α * w / (Mdot * K)`
- 新: `2 * π * α * arealpressure / (accrate * apresstoadens)`
- **结论**: ✅ **完全等价** (w=arealpressure, K=apresstoadens, Mdot=accrate)

#### **第三项对比** (term3 vs d3):

- 旧: `(1 + η) * Mdot * (3γ₁ - 1) / (2 * π * α * w * R² * 2η * (γ₃ - 1))`
- 新: `(1 + coff_eta) * accrate * (3 * chandindex_1 - 1) / (2 * π * α * arealpressure * radius² * 2 * coff_eta * (chandindex_3 - 1))`
- **结论**: ✅ **完全等价** (所有变量都是对应的重命名)

### **part2 计算结构检查结果**:

✅ **无问题发现** - 三项计算完全等价，只是变量重命名

### **u4 项检查**:

- 旧: `u4 = Qrad / (K * Mdot)` = `4πRFz / (K * Mdot)`
- 新: `u4 = energy_qrad / apresstoadens / accrate` = `4πRFz / (K * Mdot)`
- **结论**: ✅ **完全等价**

### **微分方程总结**:

经过详细检查，**微分方程的实现在数学上完全等价**，不是问题来源。

---

### 检查 22: `intial_value(r_0, lin)` vs `get_slim_initvalue(par)`

#### 旧实现

```python
def intial_value(self, r_0, lin):
    R_0 = r_0 * self.const.Rs
    l_0 = (self.const.G * self.const.M_bh * R_0) ** 0.5 / (1 - self.const.Rs / R_0) ** (-1)
    vr = (
        -(5.4e5) * self.const.alpha**0.8 * self.const.mdot**0.3
        * r_0 ** (-0.25) * self.const.m ** (-0.2)
    )
    sigma = -self.const.Mdot_0 / (2 * self.const.pi * vr * R_0)
    w_0 = (
        self.const.Mdot_0 * (l_0 - lin) / 2 / self.const.pi / self.const.alpha / R_0**2
    )
    eta_0 = vr**2 * sigma / w_0
    return np.array([l_0, eta_0])
```

#### 新实现

```python
@staticmethod
def get_slim_initvalue(*, par: DiskParams) -> np.ndarray:
    standard_result = StandardDisk.get_standard_solve_result(par=par, dimless_radius=par.dimless_radius_out)
    angmom_init = standard_result["angmom"]
    coff_eta_init = standard_result["coff_eta"]
    slim_init_arr = np.array([angmom_init, coff_eta_init])
    return slim_init_arr
```

**检查状态**: ⚠️ **完全不同的方法**
**关键差异**:

1. **旧实现**: 使用经验公式计算初始径向速度，然后反推初值
2. **新实现**: 直接使用标准薄盘解作为初值
3. **这可能导致不同的收敛行为**

---

### 检查 23: `lin(lint)` vs `get_slim_angmomin(par, dimless_angmomin)`

#### 旧实现

```python
def lin(self, lint):
    return lint * self.const.Rs * self.const.c
```

#### 新实现

```python
@staticmethod
def get_slim_angmomin(*, par: DiskParams, dimless_angmomin: float) -> float:
    radius_sch = DiskTools.get_radius_sch(par=par)
    slim_angmomin = dimless_angmomin * radius_sch * cgs_consts.cgs_c
    return slim_angmomin
```

**检查状态**: ✅ **基本正确**
**潜在问题**: 无明显问题，公式一致：`lint * Rs * c`

---

## 🚨 **核心问题发现总结**

### 最关键的发现:

1. **检查 21 - 微分方程**: ✅ **已确认无问题** - 经过详细检查，`part2` 和 `u4` 的计算在数学上完全等价
2. **检查 22 - 初值计算**: ⚠️ **完全不同的初值计算方法，这是最可能的问题来源！**

### **🎯 主要 Bug 来源**:

**初值计算方法的差异很可能是导致 vr/cs 无法超过 0.9 的根本原因！**

- **旧实现**: 使用经验公式 `vr = -5.4e5 * α^0.8 * ṁ^0.3 * r^(-0.25) * m^(-0.2)` 计算初始径向速度，然后反推初值
- **新实现**: 直接使用标准薄盘解作为初值
- **影响**: 不同的初值会导致完全不同的积分路径和收敛行为，这很可能是新求解器无法让 vr/cs 超过 0.9 的根本原因

### **其他潜在问题**:

- ⚠️ **检查 8**: 半高度计算的系数差异（待单元测试验证）
- ⚠️ **检查 20**: 速度比计算的实现细节略有不同（可能不重要）

---

## 🔧 **Bug 修复尝试**

### **已实现的解决方案**:

为了测试初值计算差异是否是问题根源，我在新求解器中添加了使用旧逻辑的初值计算方法：

#### **新增方法**:

1. **`get_slim_initvalue_legacy(par, angmomin)`** - 使用旧求解器的经验公式逻辑计算初值
2. **`slim_disk_integrator_legacy(par, angmomin)`** - 使用 legacy 初值计算的积分器
3. **`slim_disk_solver_legacy(par)`** - 使用 legacy 初值计算的完整求解器

#### **测试方法**:

```python
# 使用原始方法（可能有问题）
result_original, info_original = SlimDisk.slim_disk_solver(par=par)

# 使用 legacy 方法（应该能跨越声速）
result_legacy, info_legacy = SlimDisk.slim_disk_solver_legacy(par=par)
```

#### **预期结果**:

- 如果 legacy 方法能让 `rveltosvel_max >= 1`，则确认初值计算是问题根源
- 如果 legacy 方法仍然无法跨越声速，则需要寻找其他问题来源

**等待测试结果反馈...**
