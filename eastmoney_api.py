"""
东方财富原生API — 绕过代理直接调用，不依赖AKShare
"""
import requests, json, time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 创建不经过代理的 session
session = requests.Session()
session.trust_env = False  # 关键：绕过系统代理
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://data.eastmoney.com/'
})

EASTMONEY_HOSTS = [
    'push2.eastmoney.com', 'push2his.eastmoney.com',
    'datacenter.eastmoney.com', 'data.eastmoney.com',
    '79.push2.eastmoney.com', '17.push2.eastmoney.com',
    '83.push2.eastmoney.com', '54.push2.eastmoney.com'
]


def _get(url, params=None, retry=2):
    """带重试的HTTP GET"""
    for attempt in range(retry + 1):
        try:
            resp = session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp
        except Exception as e:
            if attempt < retry:
                time.sleep(1)
                continue
            raise
    return None


def get_limit_up_pool(date_str=None):
    """涨停板池 — 原生东财API"""
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    url = 'https://push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': '1', 'pz': '500', 'po': '1', 'np': '1',
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'fltt': '2', 'invt': '2',
        'fid': 'f3', 'fs': f'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048',
        'fields': 'f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152',
        '_': str(int(time.time() * 1000))
    }
    resp = _get(url, params)
    if resp is None: return pd.DataFrame()

    data = resp.json().get('data')
    if data is None: return pd.DataFrame()

    rows = data.get('diff', [])
    if not rows: return pd.DataFrame()

    col_map = {
        'f2': '最新价', 'f3': '涨跌幅', 'f4': '涨跌额', 'f5': '成交量',
        'f6': '成交额', 'f7': '振幅', 'f8': '换手率', 'f9': '市盈率',
        'f10': '量比', 'f12': '代码', 'f14': '名称',
        'f15': '最高', 'f16': '最低', 'f17': '开盘', 'f18': '昨收',
        'f20': '总市值', 'f21': '流通市值', 'f23': '市净率',
        'f24': '60日涨跌幅', 'f25': '5日涨跌幅', 'f26': '上市日期',
        'f62': '主力净流入', 'f115': '市盈率动态',
        'f128': '所属行业', 'f136': '市场类型', 'f152': '板块'
    }

    result = []
    for r in rows:
        row = {}
        for k, v in col_map.items():
            row[v] = r.get(k, None)
        result.append(row)
    return pd.DataFrame(result)


def get_kline(code, market='sz', days=300):
    """个股日K线 — 原生东财API"""
    secid = f'0.{code}' if code.startswith('6') else f'1.{code}'
    if market == 'sh' or code.startswith('6'):
        secid = f'1.{code}'

    end = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=days + 20)).strftime('%Y%m%d')

    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': secid,
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',  # daily
        'fqt': '1',   # 前复权
        'beg': start, 'end': end,
        'lmt': '500',
        '_': str(int(time.time() * 1000))
    }
    resp = _get(url, params)
    if resp is None: return pd.DataFrame()

    data = resp.json().get('data')
    if data is None: return pd.DataFrame()

    klines = data.get('klines', [])
    if not klines: return pd.DataFrame()

    rows = []
    for line in klines:
        parts = line.split(',')
        if len(parts) >= 11:
            rows.append({
                'date': parts[0],
                'open': float(parts[1]),
                'close': float(parts[2]),
                'high': float(parts[3]),
                'low': float(parts[4]),
                'volume': float(parts[5]),
                'amount': float(parts[6]),
                'amplitude': float(parts[7]),
                'pctChg': float(parts[8]),
                'chg': float(parts[9]),
                'turnover': float(parts[10])
            })
    return pd.DataFrame(rows)


def get_concept_fund_flow():
    """概念资金流 — 原生东财API"""
    url = 'https://push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': '1', 'pz': '500', 'po': '1', 'np': '1',
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'fltt': '2', 'invt': '2',
        'fid': 'f62', 'fs': 'm:90+t:3+f:!50',
        'fields': 'f2,f3,f4,f12,f14,f62,f66,f69,f72,f75,f78,f81,f84,f87,f104,f105,f128,f136',
        '_': str(int(time.time() * 1000))
    }
    resp = _get(url, params)
    if resp is None: return pd.DataFrame()

    data = resp.json().get('data')
    if data is None: return pd.DataFrame()

    rows = data.get('diff', [])
    if not rows: return pd.DataFrame()

    result = []
    for r in rows:
        result.append({
            '行业': r.get('f14', ''),
            '涨跌幅': r.get('f3', 0),
            '主力净流入': r.get('f62', 0),
            '超大单净流入': r.get('f66', 0),
            '大单净流入': r.get('f72', 0),
            '中单净流入': r.get('f78', 0),
            '小单净流入': r.get('f84', 0),
            '公司家数': r.get('f104', 0),
            '领涨股': r.get('f128', ''),
        })
    return pd.DataFrame(result)


def get_concept_stocks(concept_code):
    """概念板块成分股"""
    url = 'https://push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': '1', 'pz': '100', 'po': '1', 'np': '1',
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'fltt': '2', 'invt': '2',
        'fid': 'f12',
        'fs': f'b:{concept_code}+f:!50',
        'fields': 'f12,f14',
        '_': str(int(time.time() * 1000))
    }
    resp = _get(url, params)
    if resp is None: return []

    data = resp.json().get('data')
    if data is None: return []

    return [r.get('f12', '') for r in data.get('diff', [])]


def get_all_concept_codes():
    """获取所有概念板块代码"""
    url = 'https://push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': '1', 'pz': '500', 'po': '1', 'np': '1',
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'fltt': '2', 'invt': '2',
        'fid': 'f12',
        'fs': 'm:90+t:3+f:!50',
        'fields': 'f12,f14',
        '_': str(int(time.time() * 1000))
    }
    resp = _get(url, params)
    if resp is None: return {}

    data = resp.json().get('data')
    if data is None: return {}

    concept_map = {}
    for r in data.get('diff', []):
        concept_map[r.get('f14', '')] = r.get('f12', '')
    return concept_map


# ============ 测试 ============
if __name__ == '__main__':
    print('测试东方财富原生API...')

    print('\n1. 涨停池:')
    df = get_limit_up_pool()
    print(f'   {len(df)} 只涨停')
    if not df.empty:
        print(f'   列: {df.columns.tolist()}')
        print(f'   样例: {df[["代码","名称","所属行业"]].head(3).to_string()}')

    print('\n2. K线:')
    df = get_kline('000001', days=10)
    print(f'   {len(df)} 条K线')
    if not df.empty:
        print(df.tail(3).to_string())

    print('\n3. 概念资金流:')
    df = get_concept_fund_flow()
    print(f'   {len(df)} 个概念')
    if not df.empty:
        print(df.head(3).to_string())

    print('\n4. 概念板块代码:')
    codes = get_all_concept_codes()
    print(f'   {len(codes)} 个概念')
    sample = list(codes.items())[:5]
    for name, cid in sample:
        stocks = get_concept_stocks(cid)
        print(f'   {name}({cid}): {len(stocks)}只成分股')
