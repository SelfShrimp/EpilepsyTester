# Программный комплекс для обнаружения опасных кадров для эпилептиков
Данная работа представляет из себя нейросеть для обнаружения опасных кадров, метода обнаружение резкого изменения контраста и сервер, для того чтобы можно было иметь нейросеть на одном устройстве. 

Проект является частью магистерской выпускной квалификационной работы по образовательной программе "Технологии разработки компьютерных игр" [Школы разработки видеоигр Университета ИТМО](https://itmo.games/).
# Обзор
Данный программный комплекс позволяет развернуть сервер с тестированием кадров опасных для эпилептиков.  
В файле /ml/model.py вы можете настроить ссылки для запросов к определенным функциям.  
В файле /ml/views.py реализация функций, с которыми взаимодействует клиент путем POST запросов.  
Каталог /ml/modelTF содержит необходимый код для создания модели, тестовый набор данных и саму модель, если вы хотите обучить модель с нуля, то удалите каталог /ml/modelTF/model .  
# Необходимые библиотеки
```bash
pip install django tensorflow PIL numpy cv2 
```
# Пример использования клиента
```c#
using System;
using System.Diagnostics;
using System.Net.Http;
using System.Reflection;
using System.Threading.Tasks;

using (HttpClient client = new HttpClient())
using (MultipartFormDataContent form = new MultipartFormDataContent())
{
    string currentDirectory = Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName);
    // Добавление файлов к форме
    form.Add(new ByteArrayContent(System.IO.File.ReadAllBytes("Путь до\\image1.jpg")), "image1", "image1.jpg");
    form.Add(new ByteArrayContent(System.IO.File.ReadAllBytes("\\image2.jpg")), "image2", "image2.jpg");

    // Отправка POST запроса
    HttpResponseMessage response = await client.PostAsync("http://127.0.0.1/test/", form);

    // Обработка ответа от сервера
    if (response.IsSuccessStatusCode)
    {
        string responseBody = await response.Content.ReadAsStringAsync();
        Console.WriteLine("Ответ от сервера: " + responseBody);
    }
    else
    {
        Console.WriteLine("Ошибка при загрузке файлов. Код ошибки: " + response.StatusCode);
    }
    Console.ReadKey();
}
```
# Результат работы
![Alt Text](https://github.com/SelfShrimp/EpilepsyTester/blob/main/ml/modelTF/res/1.gif)
![Alt Text](https://github.com/SelfShrimp/EpilepsyTester/blob/main/ml/modelTF/res/2.gif)
![Alt Text](https://github.com/SelfShrimp/EpilepsyTester/blob/main/ml/modelTF/res/3.gif)
