import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from statistics import median
from math import trunc


detectors_amount = 0  # количество детекторов, глобальная переменная
radius = -1  # радиус каждого детектора
gray_img = None  # первый фрейм в оттенках серого?
detectors_coords = [[], []]


def get_frame_number(elem: str) -> int:
    """функция для вычленения номера фрейма из названия файла, используем для сортировки"""
    return int(elem.replace('frame', '').replace('.png', ''))


def click_event(event, x, y, flags, params):
    global detectors_amount, gray_img
    global detectors_coords
    radius = params
    if event == cv2.EVENT_LBUTTONDOWN:
        if detectors_amount > 8:
            print('maximum num of square')
            cv2.destroyAllWindows()
        else:
            # print('x =', x, ' ', 'y =', y)
            detectors_coords[0].append(x)
            detectors_coords[1].append(y)

            gray_img = cv2.rectangle(gray_img, (x - radius, y - radius), (x + radius, y + radius), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(gray_img, 'det' + str(detectors_amount) + ':' + str(x) + ',' +
                        str(y), (x + radius + 2, y), font,
                        0.8, (255, 255, 255), 2)    # 0.5, (255, 255, 255), 1)
            detectors_amount += 1


def set_detectors(radius=20):
    global gray_img
    gray_img = cv2.imread('frames/frame0.png', cv2.IMREAD_GRAYSCALE)  # первый фрейм для отметки детекторов
    # print(gray_img.shape)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.imshow('Select region', gray_img)
    cv2.namedWindow('Select region', cv2.WND_PROP_FULLSCREEN)     # cv2.WINDOW_NORMAL
    cv2.setWindowProperty('Select region', cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback('Select region', click_event, radius)
    while True:
        cv2.imshow('Select region', gray_img)
        k = cv2.waitKey(20)
        if k != -1:      # любая клавиша  было if k == ord('q'):
            break
    cv2.imwrite('frame_with_detectors.png', gray_img)
    cv2.destroyAllWindows()
    return detectors_coords


def calculate_metrics_and_log(detectors_coords: list, radius=20):
    """ Считаем средние цвета по всем детекторам по всем кадрам, после считаем наличие автомобиля на каждом кадре,
    результаты заносим в файлы. На вход принимается массив координат детекторов вида [[x1, x2, ..], [y1, y2, ..]] """
    if len(detectors_coords[0]) != len(detectors_coords[1]):
        raise Exception('У некоторых детекторов частично отсутствуют координаты')

    detectors_amount = len(detectors_coords[0])

    grey_for_mid_color_images_unsorted = os.listdir('frames')
    grey_for_mid_color_images = sorted(grey_for_mid_color_images_unsorted, key=get_frame_number)

    list_mid_color = []  # средний цвет на каждом кадре для каждого детектора
    # считаем средний цвет на каждом кадре и заносим в список
    for i in range(detectors_amount):
        temp_list_mid_color = []
        for mid_color_img in grey_for_mid_color_images:
            img = cv2.imread('frames/%s' % mid_color_img, cv2.IMREAD_GRAYSCALE)
            colors_sum = 0
            for x_pix in range(detectors_coords[0][i] - radius, detectors_coords[0][i] + radius):
                for y_pix in range(detectors_coords[1][i] - radius, detectors_coords[1][i] + radius):
                    colors_sum += img[y_pix, x_pix]
            mid_color = round(colors_sum / (4 * radius * radius), 2)
            temp_list_mid_color.append(mid_color)
        list_mid_color.append(temp_list_mid_color)
    print('> Список средних цветов посчитан')

    object_binary_list = []  # список, определяющий наличие объекта в детекторе (0 или 1)
    for i in range(0, len(list_mid_color)):  # берём списки цветов
        object_binary_temp = []
        detection_list = list_mid_color[i]
        mediana = median(detection_list)
        for det_color in detection_list:
            if abs(det_color - mediana) / mediana > 0.07:  # порог допущения 4%, если отличие в цвете меньше, то это считается погрешностью камеры
                object_binary_temp.append(1)
            else:
                object_binary_temp.append(0)
        object_binary_list.append(object_binary_temp)

    # считаем вариацию функции
    '''variation_list = []
    for det_mid_colors in list_mid_color:
        variation = 0
        for i in range(1, len(det_mid_colors)):
            variation += abs(det_mid_colors[i] - det_mid_colors[i-1])
        variation_list.append(variation)
    for var in range(len(variation_list)):
        print(f'Вариация функции {var} детектора = {variation_list[var]}')'''

    '''# считаем данные для ступенчатого графика
    for det in object_binary_list:
        for i in range(1,21):
            for frame in range((i - 1)*900 + 2, i*900):
                if det[i-2]==0 and det[i-1]==0 and det[]'''

    # пишем полные данные в файл (средний цвет)
    with open('mid_color.txt', 'w') as f:
        for item in list_mid_color:
            f.write("%s\n" % item)
    # пишем полные данные в файл (бинарный список)
    with open('binary_list.txt', 'w') as f:
        for item in object_binary_list:
            f.write(f"{item}\n")

    # пишем таблицу 0 или 1 в зависимости от наличия автомобиля
    with open('is_car_on_frame.txt', 'w') as f:
        f.write("№ Кадра |")
        for detector_number in range(detectors_amount):
            f.write("Детектор {}|".format(detector_number))
        f.write("\n")
        for i in range(len(object_binary_list[0])):
            f.write("{:4}\t|".format(i))
            for j in range(len(object_binary_list)):
                # f.write("{:3}\t|\t{:1}\n".format(i, frame))
                f.write("{:10}|".format(object_binary_list[j][i]))
                # i += step
            f.write("\n")
    return [list_mid_color, object_binary_list]


def draw_mid_color_plot(mid_color_list : list):
    x = list(range(1, len(mid_color_list[0]) + 1))
    max_value = 0
    for det_mid_col in mid_color_list:
        if max(det_mid_col) > max_value:
            max_value = max(det_mid_col)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_ylim((0, max_value))  #    axs[0, 0].set_ylim((0, max_value))

    axs[0, 0].grid()
    axs[0, 1].set_ylim((0, max_value))
    axs[0, 1].grid()
    axs[1, 0].set_ylim((0, max_value))
    axs[1, 0].grid()
    axs[1, 1].set_ylim((0, max_value))
    axs[1, 1].grid()
    axs[0, 0].plot(x, mid_color_list[0])
    # axs[0, 0].set_title('Детектор 0')
    axs[0, 1].plot(x, mid_color_list[1], c='orange')
    # axs[0, 1].set_title('Детектор 1')
    axs[1, 0].plot(x, mid_color_list[2], c='green')
    # axs[1, 0].set_title('Детектор 2')
    axs[1, 1].plot(x, mid_color_list[3], c='red')
    # axs[1, 1].set_title('Детектор 3')

    #for ax in axs.flat:
        #ax.set(xlabel='номер кадра', ylabel='цвет детектора')
    plt.show()


def draw_binary_plot(binary_lists: list):
    x = list(range(1, len(binary_lists[0]) + 1))
    fig, axs = plt.subplots(2, 2)
    # axs[0,0].set_ylim((65,120))
    # axs[0,1].set_ylim((65,120))
    # axs[1,0].set_ylim((65,120))
    # axs[1,1].set_ylim((65,120))
    axs[0, 0].plot(x, binary_lists[0])
    axs[0, 0].grid()
    # axs[0, 0].set_title('Детектор 0')
    axs[0, 1].plot(x, binary_lists[1], c='orange')
    axs[0, 1].grid()
    # axs[0, 1].set_title('Детектор 1')
    axs[1, 0].plot(x, binary_lists[2], c='green')
    axs[1, 0].grid()
    # axs[1, 0].set_title('Детектор 2')
    axs[1, 1].grid()
    axs[1, 1].plot(x, binary_lists[3], c='red')
    # axs[1, 1].set_title('Детектор 3')
    for ax in axs.flat:
        ax.set(xlabel='номер кадра', ylabel='наличие авто')
    plt.show()


def draw_quantity_plots(object_binary_list : list, dt : float, FPS : float):
    """Рисуем 2 графика - интенсивности (т.е. кол-во автомобилей в штуках) и динамики интенсивности
    (т. е. кол-во автомобилей на промежутке + предыдущ промежуток + следующий промежуток)"""
    step = trunc(dt * FPS)  # шаг берём как количество кадров в dt секундах

    # считаем данные для ступенчатого графика (количество автомобилей)
    step_plot_x = []  # список х-координат для каждого отрезка для каждого детектора
    step_plot_y = []
    for det in object_binary_list:
        tmp_x = [0]  # добавляем первое значение 0-0 (нужно для графика)
        tmp_y = [0]
        for numb in range(step, len(det), step):
            vehicle_amount = 0
            if det[numb - step:numb - step + 2] == [1, 1]:  # проверяем начало списка, есть ли там сразу автомобиль
                vehicle_amount += 1
            for frame_num in range(numb - step + 3,
                                   numb):  # идём циклом по каждому кадру и смотрим его предыдущих "соседей"
                if det[frame_num - 3: frame_num + 1] == [0, 0, 1, 1]:
                    vehicle_amount += 1
            tmp_x.append(round(numb / FPS))
            tmp_y.append(vehicle_amount)
        # для оставшейся части кадров (последней)
        vehicle_amount = 0
        if det[numb: numb + 2] == [1, 1]:  # проверяем начало списка, есть ли там сразу автомобиль
            vehicle_amount += 1
        for frame_num in range(numb + 3,
                               len(det)):  # идём циклом по каждому кадру и смотрим его предыдущих "соседей"
            if det[frame_num - 3: frame_num + 1] == [0, 0, 1, 1]:
                vehicle_amount += 1
        tmp_x.append(round(len(det) / FPS))
        tmp_y.append(vehicle_amount)
        step_plot_x.append(tmp_x)
        step_plot_y.append(tmp_y)

    # строим ступенчатый график
    fig, axs = plt.subplots(2, 2)
    # ax = fig.add_subplot()
    plt.sca(axs[0, 0])  # выбираем график (т е оси)
    plt.xticks(rotation=35)
    axs[0, 0].grid()
    axs[0, 0].step(step_plot_x[0], step_plot_y[0])

    plt.sca(axs[0, 1])  # выбираем график (т е оси)
    plt.xticks(rotation=35)
    axs[0, 1].grid()
    axs[0, 1].step(step_plot_x[1], step_plot_y[1], c='orange')

    plt.sca(axs[1, 0])  # выбираем график (т е оси)
    plt.xticks(rotation=35)
    axs[1, 0].grid()
    axs[1, 0].step(step_plot_x[2], step_plot_y[2], c='green')

    plt.sca(axs[1, 1])  # выбираем график (т е оси)
    plt.xticks(rotation=35)
    axs[1, 1].grid()
    axs[1, 1].step(step_plot_x[3], step_plot_y[3], c='red')

    max_value = 0  # вычисляем max для ylim и yticks для графика
    for det in step_plot_y:
        if max(det) > max_value:
            max_value = max(det)

    # ставим параметры для каждого графика сразу
    plt.setp(axs, xticks=np.arange(0, round(len(object_binary_list[0]) / FPS), dt), yticks=range(0, max_value + 2, 2))
    plt.setp(axs, xlim=(0, round(len(object_binary_list[0]) / FPS)), ylim=(0, max_value + 1))
    fig.tight_layout()
    plt.show()

    # строим оптимизированный график частоты
    optim_quant_plot = []
    for det in step_plot_y:
        optim_quant_plot_tmp = []
        for i in range(2, len(det) - 1):
            optim_quant_plot_tmp.append(det[i - 1] + det[i] + det[i + 1])
            # print((i - 1 ) *900, i* 900, vehicle_amount, sep=', ')
        optim_quant_plot.append(optim_quant_plot_tmp)

    max_value = 0  # вычисляем max для ylim и yticks для графика
    for det in optim_quant_plot:
        if max(det) > max_value:
            max_value = max(det)

    # строим графики, основанные на ступенчатых
    # графики строятся в секундах
    fig, axs = plt.subplots(2, 2)
    # plt.xticks(range(60,570,30))
    plt.sca(axs[0, 0])
    plt.xticks(rotation=35)
    axs[0, 0].set_ylim((0, max(optim_quant_plot[0])))
    axs[0, 0].grid()
    axs[0, 0].plot(np.arange(step_plot_x[0][2], step_plot_x[0][-1], step_plot_x[0][2] - step_plot_x[0][1]),
                   optim_quant_plot[0])  # шаг берём стандартный, макс значение на 1 шаг меньше, нач значение i0+2

    plt.sca(axs[0, 1])
    plt.xticks(rotation=35)
    axs[0, 1].set_ylim((0, max(optim_quant_plot[1])))
    axs[0, 1].grid()
    axs[0, 1].plot(np.arange(step_plot_x[0][2], step_plot_x[0][-1], step_plot_x[0][2] - step_plot_x[0][1]),
                   optim_quant_plot[1], c='orange')

    plt.sca(axs[1, 0])
    plt.xticks(rotation=35)
    axs[1, 0].set_ylim((0, max(optim_quant_plot[2])))
    axs[1, 0].grid()
    axs[1, 0].plot(np.arange(step_plot_x[0][2], step_plot_x[0][-1], step_plot_x[0][2] - step_plot_x[0][1]),
                   optim_quant_plot[2], c='green')

    plt.sca(axs[1, 1])
    plt.xticks(rotation=35)
    axs[1, 1].set_ylim((0, max(optim_quant_plot[3])))
    axs[1, 1].grid()
    axs[1, 1].plot(np.arange(step_plot_x[0][2], step_plot_x[0][-1], step_plot_x[0][2] - step_plot_x[0][1]),
                   optim_quant_plot[3], c='red')
    # ставим параметры для каждого графика сразу
    plt.setp(axs, xticks=np.arange(step_plot_x[0][2], step_plot_x[0][-1], step_plot_x[0][2] - step_plot_x[0][1]),
             yticks=range(0, 52, 2), ylim=(0, max_value + 1))

    fig.tight_layout()  # более плотное расположение (без белых полей)
    plt.show()


def get_coords_from_file(file, lines: list):
    detectors_coords = [[lines[2::2]], lines[3::2]]  # дубль глобального
