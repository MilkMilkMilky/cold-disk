# Slim Disk Solver 新旧实现对比分析

## 概述

本文档对比分析了两个 slim 吸积盘求解器的实现：

- **旧实现**: `slim_swind.py` - 使用传统的面向对象设计
- **新实现**: `solve_slim.py` - 使用现代 Python 语法和静态方法设计

## 1. 整体架构对比

### 旧实现 (`slim_swind.py`)

- **主类**: `functions` (小写类名，不符合 PEP8 规范)
- **设计模式**: 实例方法，状态存储在实例中
- **参数系统**: 使用 `const` 对象存储所有参数
- **求解函数**: 独立的 `solve()` 函数

### 新实现 (`solve_slim.py`)

- **主类**: `SlimDisk` (符合 PEP8 规范)
- **设计模式**: 静态方法，无状态设计
- **参数系统**: 使用 `DiskParams` 数据类和 `cgs_consts` 常量
- **求解函数**: 类的静态方法 `slim_disk_solver()`

## 2. 参数和常量映射表

### 2.1 物理常量映射

| 旧实现 (`const.*`) | 新实现 (`cgs_consts.*`) | 物理含义            |
| ------------------ | ----------------------- | ------------------- |
| `const.G`          | `cgs_consts.cgs_gra`    | 引力常数            |
| `const.c`          | `cgs_consts.cgs_c`      | 光速                |
| `const.a`          | `cgs_consts.cgs_a`      | 辐射常数            |
| `const.sb`         | `cgs_consts.cgs_sb`     | 斯特藩-玻尔兹曼常数 |
| `const.Rg`         | `cgs_consts.cgs_rg`     | 摩尔气体常数        |
| `const.mu`         | `cgs_consts.cgs_amm`    | 平均摩尔质量        |
| `const.kes`        | `cgs_consts.cgs_kes`    | 电子散射不透明度    |
| `const.pi`         | `math.pi`               | 圆周率              |

### 2.2 模型参数映射

| 旧实现 (`const.*`) | 新实现 (`par.*`)                | 物理含义         |
| ------------------ | ------------------------------- | ---------------- |
| `const.alpha`      | `par.alpha_viscosity`           | 粘滞参数         |
| `const.mdot`       | `par.dimless_accrate`           | 无量纲吸积率     |
| `const.M_bh`       | `DiskTools.get_bhmass(par)`     | 黑洞质量         |
| `const.Rs`         | `DiskTools.get_radius_sch(par)` | 史瓦西半径       |
| `const.N`          | `par.gas_index`                 | 气体指数         |
| `const.s`          | `par.wind_index`                | 风指数           |
| `const.r_0`        | `par.dimless_radius_out`        | 外边界无量纲半径 |
| `const.r_end`      | `par.dimless_radius_in`         | 内边界无量纲半径 |

### 2.3 辅助系数映射

| 旧实现      | 新实现                                            | 物理含义          |
| ----------- | ------------------------------------------------- | ----------------- |
| `const.IN`  | `DiskTools.get_coeff_in(index=par.gas_index)`     | 积分系数 I_N      |
| `const.IN1` | `DiskTools.get_coeff_in(index=par.gas_index + 1)` | 积分系数 I\_{N+1} |

## 3. 变量名映射表

### 3.1 主要物理量

| 旧实现变量名 | 新实现变量名      | 希腊字母/数学表示 | 物理含义         | 单位        |
| ------------ | ----------------- | ----------------- | ---------------- | ----------- |
| `R`          | `radius`          | $R$               | 半径             | cm          |
| `r`          | `dimless_radius`  | $r$               | 无量纲半径       | -           |
| `l`          | `angmom`          | $\ell$            | 角动量           | cm²/s       |
| `lin`        | `angmomin`        | $\ell_{\rm in}$   | 内边界角动量     | cm²/s       |
| `eta`        | `coff_eta`        | $\eta$            | η 系数           | -           |
| `w`          | `arealpressure`   | $w$               | 面压力           | g/s²        |
| `sigma`      | `arealdensity`    | $\Sigma$          | 面密度           | g/cm²       |
| `H`          | `halfheight`      | $H$               | 半高度           | cm          |
| `P`          | `pressure`        | $P$               | 压力             | g/(cm·s²)   |
| `rho`        | `density`         | $\rho$            | 密度             | g/cm³       |
| `T`          | `temperature`     | $T$               | 温度             | K           |
| `Teff`       | `temperature_eff` | $T_{\rm eff}$     | 有效温度         | K           |
| `kappa`      | `opacity`         | $\kappa$          | 不透明度         | cm²/g       |
| `beta`       | `pressure_ratio`  | $\beta$           | 压力比           | -           |
| `gamma1`     | `chandindex_1`    | $\Gamma_1$        | 钱德拉塞卡指数 1 | -           |
| `gamma3`     | `chandindex_3`    | $\Gamma_3$        | 钱德拉塞卡指数 3 | -           |
| `Fz`         | `fluxz`           | $F_z$             | 垂直通量         | erg/(cm²·s) |
| `vr`         | `radvel`          | $v_r$             | 径向速度         | cm/s        |
| `cs`         | `soundvel`        | $c_s$             | 声速             | cm/s        |
| `K`          | `apresstoadens`   | $K$               | 面压力与面密度比 | cm²/s²      |

### 3.2 导数和微分量

| 旧实现变量名   | 新实现变量名         | 希腊字母/数学表示         | 物理含义       |
| -------------- | -------------------- | ------------------------- | -------------- |
| `omega_k`      | `angvelk`            | $\Omega_K$                | 开普勒角速度   |
| `lk`           | `angmomk`            | $\ell_K$                  | 开普勒角动量   |
| `Mdot`         | `accrate`            | $\dot{M}$                 | 吸积率         |
| `dlnMdot_dr`   | `dlnaccrate_dradius` | $\frac{d\ln\dot{M}}{dr}$  | 吸积率对数导数 |
| `dlnomegak_dr` | `dlnangvelk_dradius` | $\frac{d\ln\Omega_K}{dr}$ | 角速度对数导数 |
| `dl_dR`        | `dangmom_dradius`    | $\frac{d\ell}{dR}$        | 角动量导数     |
| `deta_dR`      | `dcoffeta_dradius`   | $\frac{d\eta}{dR}$        | η 系数导数     |

### 3.3 求解器变量

| 旧实现变量名 | 新实现变量名           | 希腊字母/数学表示        | 含义               |
| ------------ | ---------------------- | ------------------------ | ------------------ |
| `lint`       | `dimless_angmomin`     | $\tilde{\ell}_{\rm in}$  | 无量纲内边界角动量 |
| `lintmin`    | `dimless_angmomin_min` | $\tilde{\ell}_{\rm min}$ | 搜索下界           |
| `lintmax`    | `dimless_angmomin_max` | $\tilde{\ell}_{\rm max}$ | 搜索上界           |
| `vr_vs`      | `rveltosvel`           | $v_r/c_s$                | 径向速度与声速比   |
| `t_array`    | `indep_array`          | $\{r_i\}$                | 积分变量数组       |
| `y0`         | `initvalue`            | $[\ell_0, \eta_0]$       | 初值               |

## 4. 方法映射表

### 4.1 基本物理量计算方法

| 旧实现方法                   | 新实现方法                                                      | 功能               |
| ---------------------------- | --------------------------------------------------------------- | ------------------ |
| `get_omega_k(R)`             | `get_slim_angvelk(par, radius)`                                 | 计算开普勒角速度   |
| `get_domega_dR(R)`           | ❌ (未直接实现)                                                 | 计算角速度导数     |
| `get_Mdot(R)`                | `get_slim_accrate(par, radius)`                                 | 计算吸积率         |
| `get_dlnMdot_dr(R, Mdot)`    | `get_slim_dlnaccrate_dradius(par, radius)`                      | 计算吸积率对数导数 |
| `get_dlnomegak_dr(R)`        | `get_slim_dlnangvelk_dradius(par, radius)`                      | 计算角速度对数导数 |
| `get_w(R, l, lin, Mdot)`     | `get_slim_arealpressure(par, radius, angmom, angmomin)`         | 计算面压力         |
| `get_sigma(w, eta, R, Mdot)` | `get_slim_arealdensity(par, arealpressure, coff_eta, radius)`   | 计算面密度         |
| `get_H(omega_k, w, sigma)`   | `get_slim_halfheight(par, radius, arealpressure, arealdensity)` | 计算半高度         |
| `get_P(w, H)`                | `get_slim_pressure(par, arealpressure, halfheight)`             | 计算压力           |
| `get_rho(sigma, H)`          | `get_slim_density(par, arealdensity, halfheight)`               | 计算密度           |

### 4.2 热力学和辐射方法

| 旧实现方法                 | 新实现方法                                                  | 功能                 |
| -------------------------- | ----------------------------------------------------------- | -------------------- |
| `get_T(b, c)`              | `get_slim_temperature(pressure, density)`                   | 计算温度             |
| `get_kappa(rho, T)`        | `get_slim_opacity(par, density, temperature)`               | 计算不透明度         |
| `get_beta(rho, T, P)`      | `get_slim_pressure_ratio(density, temperature, pressure)`   | 计算压力比           |
| `get_gamma1(beta)`         | `get_slim_chandindex_1(pressure_ratio)`                     | 计算钱德拉塞卡指数 1 |
| `get_gamma3(beta)`         | `get_slim_chandindex_3(pressure_ratio)`                     | 计算钱德拉塞卡指数 3 |
| `get_Fz(rho, H, kappa, T)` | `get_slim_fluxz(density, halfheight, opacity, temperature)` | 计算垂直通量         |
| `get_K(sigma, w)`          | `get_slim_apresstoadens(arealpressure, arealdensity)`       | 计算面压力密度比     |

### 4.3 速度和运动学方法

| 旧实现方法    | 新实现方法                                   | 功能         |
| ------------- | -------------------------------------------- | ------------ |
| ❌ (内联计算) | `get_slim_radvel(par, radius, arealdensity)` | 计算径向速度 |
| ❌ (内联计算) | `get_slim_soundvel(pressure, density)`       | 计算声速     |
| ❌ (内联计算) | `get_slim_rveltosvel(radvel, soundvel)`      | 计算速度比   |
| ❌ (内联计算) | `get_slim_temperature_eff(fluxz)`            | 计算有效温度 |

### 4.4 求解器核心方法

| 旧实现方法                | 新实现方法                                           | 功能             |
| ------------------------- | ---------------------------------------------------- | ---------------- |
| `main_fuction(r, y, lin)` | `slim_disk_model(indep_var, dep_var, par, angmomin)` | 微分方程组       |
| `intial_value(r_0, lin)`  | `get_slim_initvalue(par)`                            | 计算初值         |
| `lin(lint)`               | `get_slim_angmomin(par, dimless_angmomin)`           | 转换内边界角动量 |
| `solve(const)`            | `slim_disk_solver(par)`                              | 主求解器         |

### 4.5 新实现中的额外方法

| 新实现方法                                                                       | 功能               |
| -------------------------------------------------------------------------------- | ------------------ |
| `get_slim_angmomk(par, radius)`                                                  | 计算开普勒角动量   |
| `get_slim_indep_array(par)`                                                      | 生成积分变量数组   |
| `slim_disk_integrator(par, angmomin)`                                            | 积分器封装         |
| `get_slim_rveltosvel_fromfirst(par, dimless_radius, angmom, coff_eta, angmomin)` | 从基本量计算速度比 |

## 5. Slim 吸积盘求解器模型运行思路

### 5.1 物理模型概述

Slim 吸积盘模型是对标准薄盘模型的扩展，考虑了以下物理效应：

- **径向对流**: 通过 η 系数描述径向对流效应
- **相对论效应**: 在史瓦西度规下的开普勒轨道
- **辐射压**: 考虑辐射压与气体压的混合
- **变吸积率**: 允许吸积率随半径变化（风损失）

### 5.2 核心微分方程组

求解器求解两个耦合的常微分方程：

1. **角动量方程**: `dl/dr = f₁(r, l, η)`
2. **对流系数方程**: `dη/dr = f₂(r, l, η)`

其中：

- `l(r)`: 比角动量随半径的分布
- `η(r)`: 对流系数随半径的分布
- `r`: 无量纲半径 (以史瓦西半径为单位)

### 5.3 打靶法求解策略

#### 5.3.1 边界条件问题

- **外边界** `dimless_radius_out`($r_{\rm out}$): 已知标准薄盘解作为初值
- **内边界** `dimless_radius_in`($r_{\rm in}$): 要求径向速度等于声速 ($v_r/c_s = 1$)

#### 5.3.2 打靶参数

- **目标参数**: 内边界角动量 `angmomin`($\ell_{\rm in}$) (或无量纲形式 `dimless_angmomin`($\tilde{\ell}_{\rm in}$) )
- **搜索范围**: `[dimless_angmomin_min, dimless_angmomin_max]` $[\tilde{\ell}_{\rm min}, \tilde{\ell}_{\rm max}]$
- **收敛判据**: 在内边界处实现跨声速条件

#### 5.3.3 打靶算法流程

```
1. 初始化搜索区间 [dimless_angmomin_min, dimless_angmomin_max]
2. 选择中点值 dimless_angmomin_guess
3. 从外边界积分到内边界
4. 检查收敛条件：
   - 如果积分失败 (dimless_radius_solve_min > 3): dimless_angmomin_max = dimless_angmomin_guess
   - 如果 max(rveltosvel) < 1: dimless_angmomin_min = dimless_angmomin_guess
   - 如果 max(rveltosvel) ≥ 1: 收敛成功
5. 更新搜索区间，重复步骤 2-4
```

### 5.4 物理量计算流程

#### 5.4.1 基本量计算顺序

```
r → R → Mdot → l, η → w → σ → H → P, ρ → T → κ, β → γ₁, γ₃ → Fz
```

#### 5.4.2 关键物理关系

- **面压力**: `w = Mdot(l-lin)/(2πα R²)`
- **面密度**: `σ = Mdot²/(4π² R² w η)`
- **半高度**: `H = √[(2N+2)wI_N/(σI_{N+1})] / Ωₖ`
- **温度**: 通过辐射平衡方程求解 `(a/3)T⁴ + bT + c = 0`

### 5.5 跨声速条件的物理意义

#### 5.5.1 声速屏障

- 在内边界附近，径向速度接近声速
- 这是 slim 盘与标准薄盘的重要区别
- 跨声速条件确保了解的物理自洽性

#### 5.5.2 临界角动量

- `angmomin`($\ell_{\rm in}$) 是使盘在内边界跨声速的临界角动量
- 过大的 `angmomin` 导致亚声速流
- 过小的 `angmomin` 导致积分发散

### 5.6 数值实现细节

#### 5.6.1 积分方法

- 使用 `odeint` 进行常微分方程组积分
- 从外边界向内边界积分 (`dimless_radius`($r$) 递减)
- 高精度设置: `atol=1e-10, rtol=1e-10`

#### 5.6.2 收敛监控

- 实时计算 $v_r/c_s$ 比值
- 监控积分是否到达目标内边界
- 记录所有中间物理量用于后续分析

### 5.7 模型的适用范围

#### 5.7.1 高吸积率区域

- 适用于 `ṁ ≳ 0.01` 的高吸积率情况
- 在这种情况下辐射压变得重要
- 径向对流不可忽略

#### 5.7.2 相对论区域

- 在靠近黑洞的区域 (`dimless_radius`($r$) $\lesssim 100$)
- 相对论效应显著影响轨道动力学
- 标准薄盘近似失效

## 6. 两种实现的关键差异

### 6.1 初值计算方法

#### 旧实现 (`intial_value`)

- 使用经验公式估算初始径向速度
- 基于径向速度反推初始面密度和对流系数
- 公式: `vr = -5.4e5 * α^0.8 * ṁ^0.3 * r^(-0.25) * m^(-0.2)`

#### 新实现 (`get_slim_initvalue`)

- 直接使用标准薄盘解作为初值
- 通过 `StandardDisk.get_standard_solve_result()` 获取
- 更加物理自洽的初值选择

### 6.2 求解器收敛判据

#### 旧实现

```python
if min_solve_t > 3:
    lintmax = lint
else:
    if max(fuc.vr_vs) < 1:
        lintmin = lint
    else:
        print('get solve! lint:', lint)
        break
```

#### 新实现

```python
if dimless_radius_solve_min > 3:
    dimless_angmomin_max = dimless_angmomin
else:
    if rveltosvel_max < 1:
        dimless_angmomin_min = dimless_angmomin
    else:
        # 收敛成功
        break
```

### 6.3 数据存储和输出

#### 旧实现

- 实时存储所有中间计算结果到 `logdata` 列表
- 返回结构化 NumPy 数组，包含 18 个物理量
- 便于后续分析和可视化

#### 新实现

- 仅返回求解结果 (半径、角动量、对流系数)
- 需要额外调用计算函数获取其他物理量
- 更加模块化但需要额外处理

## 7. 数值验证要点

### 7.1 关键检验点

1. **相同参数下的收敛性**: 验证两种实现是否收敛到相同的 `angmomin`($\ell_{\rm in}$) 值
2. **物理量一致性**: 对比关键物理量 (temperature($T$)、density($rho$)、fluxz($F_z$) 等) 的数值
3. **边界条件**: 确认跨声速条件 `(rveltosvel)`($v_r/c_s$) $\approx 1$ 在内边界得到满足
4. **积分精度**: 验证积分路径和精度设置的影响

### 7.2 潜在数值差异源

1. **温度求解方法**: `fsolve` vs `root_scalar` 可能产生微小差异
2. **初值差异**: 不同的初值计算方法可能影响收敛路径
3. **数组处理**: NumPy 数组操作的细微差异
4. **常数精度**: 物理常数的精度差异

## 8. 模型物理意义总结

### 8.1 Slim 盘的核心特征

1. **跨声速流动**: 内边界处径向速度达到声速，这是区别于标准薄盘的关键特征
2. **径向对流**: η 系数描述的径向对流输运，在高吸积率下变得重要
3. **辐射压主导**: 在内区辐射压超过气体压，影响盘的结构
4. **相对论效应**: 史瓦西度规下的轨道动力学修正

### 8.2 打靶法的物理必要性

- **边界值问题**: 外边界条件已知，内边界条件 (跨声速) 待确定
- **非线性耦合**: `angmom`($\ell$) 和 `coff_eta`($\eta$) 的强耦合使得解析解不可能
- **临界现象**: 跨声速条件类似于临界现象，需要精确的参数调节
- **解的唯一性**: 正确的 `angmomin`($\ell_{\rm in}$) 值确保物理解的唯一性和稳定性

### 8.3 应用前景

该求解器为研究以下天体物理现象提供了重要工具：

- **超大质量黑洞吸积**: 活动星系核和类星体的能量输出
- **恒星质量黑洞**: X 射线双星系统的高态观测
- **中等质量黑洞**: 球状星团中心的吸积现象
- **原始黑洞**: 早期宇宙中的吸积过程
