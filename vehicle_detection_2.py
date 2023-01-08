import os
import video_processing
import pathlib
import calculate_detectors
import glob
from ctypes import windll
import traceback


if __name__ == '__main__':
    try:
        #script_path = pathlib.Path(__file__).parent.resolve()
        #os.chdir(script_path)   # меняем директорию на ту, в которой лежит скрипт
        # шаг на графике интенсивности и динамики интенсивности
        plot_step_sec = int(input('Введите шаг для графика в секундах и нажмите Enter (рекомендуется 5)\n > '))
        all_frames = glob.glob('frames/*')  # удаляем все фреймы из папки
        for f in all_frames:
            os.remove(f)

        VIDEO_LINK = 'https://www.youtube.com/watch?v=ICr4egOfX10' #https://www.youtube.com/watch?v=VeFl0LPzjYU, https://www.youtube.com/watch?v=K-DNGBOCPnI

        video_processing.download_fragment(VIDEO_LINK)

        FPS = video_processing.split_video_by_frames(1)

        # проверка наличия файла с координатами детекторов
        if os.path.exists('last_execution.txt'):
            memory_file = open('last_execution.txt', 'r+')
            lines = memory_file.readlines()
            if len(lines) != 0:
                if lines[0].rstrip('\n') == VIDEO_LINK:
                    execution_is_empty = False     # файл не пуст и с требуемой ссылкой
                else:
                    memory_file.close()
                    memory_file = open("last_execution.txt", "w")
                    execution_is_empty = True     # файл не пуст, но с отличающейся ссылкой
            else:
                execution_is_empty = True
        else:
            memory_file = open("last_execution.txt", "w")
            execution_is_empty = True    # файл пуст

        if not execution_is_empty:
            detectors_coords = [int(coord.rstrip('\n')) for coord in lines[1:]] # получаем координаты детекторов списком (смешанные)
            x_coords = detectors_coords[0::2]   # отбираем координаты по x
            y_coords = detectors_coords[1::2]
            video_processing.save_preview([x_coords, y_coords], 20)
            plot_lists = calculate_detectors.calculate_metrics_and_log([x_coords, y_coords], 20)    # возвращаем списки средних цветов и наличия автомобиля в кадре
            calculate_detectors.draw_mid_color_plot(plot_lists[0])
            calculate_detectors.draw_binary_plot(plot_lists[1])
            calculate_detectors.draw_quantity_plots(plot_lists[1], plot_step_sec, FPS)
            memory_file.close()
        else:
            windll.user32.MessageBoxW(0,
                                             'Далее будет происходить расстановка детекторов. Поставьте 4 экземпляра и нажмите любую кнопку',
                                             'Инструкция', 0)
            detectors_coords = calculate_detectors.set_detectors(10)
            plot_lists = calculate_detectors.calculate_metrics_and_log(detectors_coords)
            memory_file.write(VIDEO_LINK)
            for detector_num in range(len(detectors_coords[0])):
                memory_file.write('\n' + str(detectors_coords[0][detector_num]) + '\n' + str(detectors_coords[1][detector_num]))
            memory_file.close()
            print('> Файл с данными о детекторах обновлён')
            calculate_detectors.draw_mid_color_plot(plot_lists[0])  # возвращаем списки средних цветов и наличия автомобиля в кадре
            calculate_detectors.draw_binary_plot(plot_lists[1])
            calculate_detectors.draw_quantity_plots(plot_lists[1], plot_step_sec, FPS)
    except Exception as e:
        print(f'Ошибка! {traceback.format_exc()}')
    finally:
        os.system('pause')