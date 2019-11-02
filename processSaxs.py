import numpy as np


def is_number(s):
    try:
        float(s)  # for int, long and float
    except ValueError:
        return False
    return True


def read_standardiq_dat(file_name):
    qis = 0
    try:
        file = open(file_name, 'r')
    except Exception:
        return 1, 1
    q = []
    i = []
    s = []
    comments = []
    try:
        content = file.readlines()
    except UnicodeDecodeError:
        return 2, 1
    for line in content:
        try:
            if line[0] != "#":
                word = line.strip().split()
                if (len(word) == 3 or len(word) == 4) and is_number(word[0]):
                    qis = 3
                    if float(word[0]) == 0:
                        continue
                    q.append(float(word[0]))
                    i.append(float(word[1]))
                    s.append(float(word[2]))
                elif len(word) == 2 and is_number(word[0]):
                    qis = 2
                    if float(word[0]) == 0:
                        continue
                    q.append(float(word[0]))
                    i.append(float(word[1]))
                if len(q) == 0:
                    word = line.strip().split(',')
                    if (len(word) == 3 or len(word) == 4) and is_number(word[0]):
                        qis = 3
                        if float(word[0]) == 0:
                            continue
                        q.append(float(word[0]))
                        i.append(float(word[1]))
                        s.append(float(word[2]))
                    elif len(word) == 2 and is_number(word[0]):
                        qis = 2
                        if float(word[0]) == 0:
                            continue
                        q.append(float(word[0]))
                        i.append(float(word[1]))
        except:
            pass
    q = np.array(q).reshape((-1, 1))
    i = np.array(i).reshape((-1, 1))
    s = np.array(s).reshape((-1, 1))
    while q[-1] > 1:
        q = q / 10
    if qis == 3:
        result = np.concatenate((q, i, s), axis=1)
    elif qis == 2:
        result = np.concatenate((q, i), axis=1)
    return 0, result


def read_standardiq_out(file_name):
    try:
        file = open(file_name, 'r')
    except Exception:
        return 1, 1, 1
    q = []
    i = []
    comments = []
    try:
        content = file.readlines()
    except UnicodeDecodeError:
        return 2, 1, 1

    for line in content:
        word = line.strip().split()
        if 'Real space range' in line:
            Rmax = float(word[-1]) / 2
            print Rmax
        all_digit = np.array(list(map(is_number, word)))
        if len(word) == 5 and (all_digit == True).all():
            q.append(float(word[0]))
            i.append(float(word[3]))
    q = np.array(q).reshape((-1, 1))
    i = np.array(i).reshape((-1, 1))
    while q[-1] > 1:
        q = q / 10
        Rmax = Rmax * 10
    result = np.concatenate((q, i), axis=1)
    return 0, result, Rmax


def read_iq(file_name):
    separators = ['', ' ', ',', ';', '->', '&']
    file = open(file_name, 'r')
    q = []
    i = []
    s = []
    comments = []
    for line in file:
        all_good = False
        try:
            if line[0] != "#":
                keys = line.split("\n")[0].split()
                new_keys = []
                for key in keys:
                    if key not in separators:
                        new_keys.append(key)
                keys = new_keys
                q.append(float(keys[0]))
                i.append(float(keys[1]))
            else:
                comments.append(line[0:len(line) - 1])
            all_good = True
        except:
            pass
        if not all_good:
            print "WARNING TROUBLE READING THIS LINE:"
    q = np.array(q).reshape((-1, 1))
    i = np.array(i).reshape((-1, 1))
    while q[-1] > 1:
        q = q / 10
    result = np.concatenate((q, i), axis=1)
    return result


def average_filter(data, step, derivation=False):
    size = data.shape[0]
    seed = np.linspace(20, 1, 10)
    weights = seed[:step] / (2 * sum(seed[:step]) - seed[0])
    datacopy = np.zeros((size, 2))
    if derivation == False:
        for ii in range(step - 1, size - step + 1):
            new_value = 0
            for jj in range(step):
                new_value = new_value + (data[ii - jj][1] + data[ii + jj][1]) * weights[jj]
            new_value = new_value - data[ii][1] * weights[0]
            datacopy[ii][1] = new_value
    datacopy[step - 1:size - step + 1, 0] = data[step - 1:size - step + 1, 0]
    return datacopy[step - 1:size - step + 1]


def process(file_path):
    Rmax = ''
    result = []
    if file_path.split('.')[-1] == 'out':
        stat, iq_curve, Rmax = read_standardiq_out(file_path)
        if stat != 0:
            return False
    else:
        # iq_curve = read_iq(file_path)
        stat, iq_curve = read_standardiq_dat(file_path)
        if stat != 0:
            return False
    fixiq = average_filter(iq_curve, 1).reshape((-1, 2))
    # be careful numpy is ke bian bain liang, xiangdangyu yingyong

    fitx = np.copy(fixiq[:, 0]).reshape(-1)
    fity = np.copy(fixiq[:, 1]).reshape(-1)

    if iq_curve[0, 0] // 0.005 == 0:
        xstart = 0.005
    elif iq_curve[0, 0] % 0.005 == 0:
        xstart = (iq_curve[0, 0] // 0.005) * 0.005
    else:
        xstart = (iq_curve[0, 0] // 0.005 + 1) * 0.005

    xend = 0.201 if 0.201 < iq_curve[-1, 0] else iq_curve[-1, 0]
    x_scale = np.arange(xstart, xend, 0.005, dtype=float)
    fitlen = len(x_scale)
    y_value = np.zeros(shape=(fitlen), dtype=float)
    position = 0

    for ii in range(len(fitx)):
        if fitx[ii] >= x_scale[position]:
            if ii - 2 < 0:
                y_fit = np.polyfit(fitx[:ii + 3], fity[:ii + 3], 1)
                a, b = y_fit.tolist()
                y_value[position] = a * x_scale[position] + b
                continue
            elif ii + 2 > len(fitx):
                y_fit = np.polyfit(fitx[ii - 3:], fity[ii - 3:], 2)
            else:
                y_fit = np.polyfit(fitx[ii - 2:ii + 2], fity[ii - 2:ii + 2], 2)
            a, b, c = y_fit.tolist()
            y_value[position] = a * x_scale[position] ** 2 + b * x_scale[position] + c
            position += 1
            if position == fitlen:
                break
    preprocessediq = np.concatenate((x_scale.reshape(-1, 1), y_value.reshape(-1, 1)), axis=1)
    result.append(preprocessediq)
    if file_path.split('.')[-1] == 'out':
        result.append(Rmax)
    return result


def generatesaxsstr(file_path):
    if file_path.split('.')[-1] == 'out':
        stat, iq_curve, Rmax = read_standardiq_out(file_path)
        if stat != 0:
            return False
    else:
        stat, iq_curve = read_standardiq_dat(file_path)
        if stat != 0:
            return False
    saxsstr = ''
    for ii in range(len(iq_curve)):
        saxsstr = saxsstr + '{x: %f, y: %f},' % (iq_curve[ii, 0], abs(iq_curve[ii, 1]))
    print saxsstr[-70:]
    return saxsstr