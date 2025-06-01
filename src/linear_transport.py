from scipy.optimize import linprog
import pandas as pd
import numpy as np
import os
import json

def load_data(file_path):
    '''
    input:
        file_path: 数据文件路径
    output:
        Moveout_sorted: 移出县排序 shape(len(livestock_out), ...)
        Movein_sorted: 移入县排序 shape(len(livestock_in), ...)
    '''
    Movein = pd.read_excel(file_path, sheet_name='移入')
    Moveout = pd.read_excel(file_path, sheet_name='移出')
    Move_out_sorted = pd.read_excel(file_path, sheet_name='移出排序')
    Move_out_sorted = Move_out_sorted[['idcode', '移出等级']]
    Move_in_sorted = pd.read_excel(file_path, sheet_name='移入排序')
    Move_in_sorted = Move_in_sorted[['idcode', '移入等级']]
    # 按照Move_out_sorted中的移出等级，将Moveout中的县名进行排序
    Moveout_sorted = Moveout.merge(Move_out_sorted, on='idcode', how='left')
    Moveout_sorted = Moveout_sorted.sort_values(by='移出等级', ascending=True)
    Movein_sorted = Movein.merge(Move_in_sorted, on='idcode', how='left')
    Movein_sorted = Movein_sorted.sort_values(by='移入等级', ascending=True)
    return Moveout_sorted, Movein_sorted

def solve_movement_lp(weight_out, weight_in, BNF_out, BNF_in, livestock_out):
    """
    使用线性规划解决移动量问题。
    目标: maxize \sum_{i=1}^{m} livestock_out_i * x_i
    约束：BNF_in + \sum_{j=1}^{n} livestock_out_j * x_j * weight_in_j >= 0
         BNF_out - \sum_{i=1}^{m} livestock_out_i * x_i * weight_out_i <= 0
         0 <= x_i <= 1
    """
    n = len(livestock_out)  # 决策变量的数量
    
    # 目标函数：最大化移动量（最小化负的移动量）
    c = [-1] * n  # 直接使用livestock_out作为目标函数系数
    
    # 构建约束矩阵
    # 1. 移入县约束：BNF_in + sum(livestock_out_j * x_j * weight_in_j) <= 0
    # 2. 移出县约束：BNF_out - sum(livestock_out_i * x_i * weight_out_i) >= 0
    if (weight_in == weight_out).all():
        A_ub = np.vstack([
            livestock_out * weight_out   
        ])
        b_ub = np.array([min(-BNF_in, BNF_out)])
    else:
        A_ub = np.vstack([
            livestock_out * weight_in,
            livestock_out * weight_out
        ])
        b_ub = np.array([-BNF_in, BNF_out])
    # 变量范围约束：0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(n)]
    
    try:
        # 使用 'highs' 方法求解LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x
        else:
            print(f"线性规划求解失败: {res.message}")
            return None
    except Exception as e:
        print(f"线性规划求解出错: {str(e)}")
        return None

def simulate_livestock_movement(Moveout_sorted, Movein_sorted):
    # 创建数据的副本以在函数内部操作，避免修改原始输入
    Moveout_sorted_copy = Moveout_sorted.copy()
    Movein_sorted_copy = Movein_sorted.copy()
    
    outflow_idx = 0  # 指向当前最高优先级的移出县
    inflow_idx = 0   # 指向当前最高优先级的移入县
    
    movements_log = []

    while outflow_idx < len(Moveout_sorted_copy) and inflow_idx < len(Movein_sorted_copy):
        out_county = Moveout_sorted_copy.iloc[outflow_idx]
        in_county = Movein_sorted_copy.iloc[inflow_idx]

        source_balance = out_county['BNF+manure-opn(kg)']
        target_balance = in_county['BNF+manure-opn（kg）']

        # 获取移出县的当前牲畜数量
        livestock_out = out_county.values[4:4+6]

        # 获取包含"移动"的列的值
        out_move_weights = out_county[out_county.index.str.contains('移动')]
        in_move_weights = in_county[in_county.index.str.contains('移动')]

        # 检查移出县是否已耗尽 (BNF+manure-opn(kg) 不再为负)
        source_is_depleted = (source_balance <= min(out_move_weights)) or (livestock_out == 0).all()
        if source_is_depleted:
            outflow_idx += 1  # 移至下一个候选移出县
            continue

        # 检查移入县是否已满足 (BNF+manure-opn(kg) + 最小移动系数>= 0)
        target_is_satisfied = (target_balance + min(in_move_weights) > 0)
        if target_is_satisfied:
            inflow_idx += 1  # 移至下一个候选移入县
            continue
            
        # 使用线性规划确定移动量
        moved_amount = solve_movement_lp(out_move_weights.values, 
                                      in_move_weights.values, 
                                      source_balance,
                                      target_balance,
                                      livestock_out)

        if isinstance(moved_amount, np.ndarray):
            moved_amount = moved_amount * livestock_out
            BNF_del_out = moved_amount @ out_move_weights.values
            BNF_del_in = moved_amount @ in_move_weights.values
            Moveout_sorted_copy.iloc[outflow_idx, -2] -= BNF_del_out
            Movein_sorted_copy.iloc[inflow_idx, -2] += BNF_del_in
            # 更新移动量
            Moveout_sorted_copy.iloc[outflow_idx, 4:4+6] -= moved_amount
            Movein_sorted_copy.iloc[inflow_idx, 4:4+6] += moved_amount
            
            movements_log.append({
                'from': out_county['idcode'],
                'to': in_county['idcode'],
                'amount': moved_amount,
                'delta_BNF+manure-opn(kg)_out': BNF_del_out,
                'delta_BNF+manure-opn(kg)_in': BNF_del_in
            })
            print(f"县{out_county['idcode']} -> 县{in_county['idcode']} 移动量: {moved_amount}")
        else:
            print(f"县{out_county['idcode']}和县{in_county['idcode']}无法移动")
            break

    # 保存结果
    if not os.path.exists('result'):
        os.makedirs('result')
    Moveout_sorted_copy.to_excel('result/巴西移动分县0528_out.xlsx', index=False)
    Movein_sorted_copy.to_excel('result/巴西移动分县0528_in.xlsx', index=False)
    # 保存movements_log
    def np_encoder(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open('result/巴西移动分县0528_movements_log.json', 'w', encoding='utf-8') as f:
        json.dump(movements_log, f, ensure_ascii=False, indent=2, default=np_encoder)
    return movements_log, Moveout_sorted_copy, Movein_sorted_copy

if __name__ == "__main__":
    Moveout_sorted, Movein_sorted = load_data('data/巴西移动分县0528.xlsx')
    movements_log, Moveout_sorted_copy, Movein_sorted_copy = simulate_livestock_movement(Moveout_sorted, Movein_sorted)
    print("移动完成")