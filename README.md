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
