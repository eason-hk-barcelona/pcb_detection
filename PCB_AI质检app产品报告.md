# PCB_AI质检app产品报告

## 功能介绍

质检app运行环境为HarmonyOS 4.0.0，api版本是6，采用JS FA模型开发，可针对Mouse_bite、Open_circuit、Short、Spur、Spurious_copper等瑕疵类型的pcb板进行瑕疵检测。整个 app 共分为历史速览、异常检测、历史记录三大模块

### 异常检测

质检员可以自由的对有瑕疵的pcb板进行拍摄，调用云端modelarts推理，并在首页返回推理结果图（包含该pcb板的瑕疵位置、瑕疵类型、检测置信度）

### 历史记录

质检员可以根据检测日期和图片名称（以时间戳命名）精准定位之前检测过的pcb板，每条历史记录包含该板的推理结果图，同时以饼状图的形式统计该板不同瑕疵类型占比和异常总数

### 历史速览

为方便质检员快速查询，该模块记录了历史最新5次的检测结果，以滑动窗口和缩略图的形式呈现，缩略图支持点击查看大图功能

## 程序概要设计

### 前端

#### 项目结构

```SHELL
pages
├───index
 	 index.css
	index.hml
	index.js
├───page
        page.css
        page.hml
        page.js
│───splash
        page.css
        page.hml
        page.js
```

前端由四个页面组成，分别是：启动页、异常检测、首页、历史记录，通过`@ohos.router`的router相互路由，网络请求均通过`@ohos.net.http`原生http进行promise异步请求

#### 异常检测

异常检测调用camera组件，通过其takePhoto中的complete方法返回的`result.uri`通过简单的正则拿到照片沙箱路径，创建一个原始二进制缓冲区`ArrayBuffer`，通过`@ohos.fileio`的fileIO实现图片文件同步读写，最后将读出的字节序列字符串放入请求体中传给后端

#### 首页

进度条

设立一个`isLoading`的flag，当质检员拍照，flag置为true，调用`progressLoading`启动进度条，其在 flag 为 true 时递增 `progress`组件中的`percent`，并使用 `setTimeout` 继续调用自己，实现循环更新，当`modelarts`推理返回推理结果图时flag置为false，此时首页渲染最新的检测结果图

滑动窗口

使用 `swiper` 组件实现图片的分页滑动显示，每个页面最多显示三张图片。通过 `recentImages` 数组和 `slice` 方法，动态地从数组中提取图片进行显示。

#### 历史记录

在生命周期`onInit()`方法中，使用全局变量 `globalThis.value` 来获取图片 ID，构建图片 URL，实现image标签src请求图片

维护一个`detectionClasses` 数组来存储每个异常类别的**名称**和**占比**，后端返回的推理样例如下：

```json
{
  "filename": "test.png",
  "result": {
    "detection_boxes": [
      [
        1580.695556640625,
        1063.1583251953125,
        1632.359619140625,
        1074.5572509765625
      ],
      [
        1577.9443359375,
        1063.1236572265625,
        1629.8369140625,
        1074.6903076171875
      ],
      [
        707.0120239257812,
        517.9932250976562,
        747.9051513671875,
        527.4000854492188
      ]
    ],
    "detection_classes": [
      "Spurious_copper",
      "Spurious_copper",
      "Spurious_copper"
    ],
    "detection_scores": [
      0.2881912589073181,
      0.24983583390712738,
      0.19878366589546204
    ]
  },
  "source": "ModelArts",
  "success": true
}
```

利用HarmonyOS内置的原子服务，可实现三次敲击屏幕放大，双指拖动图片，以查看检测结果图详情

### 后端

主要利用flask，postgres进行开发，实现调用华为云modelarts，数据库存入历史检测数据，根据modelarts返回的瑕疵坐标定位并绘制瑕疵框等功能

#### modelarts

首先根据modelarts官方文档构造获取token的请求函数`get_token`，将token放入req header中的X-Auth-Token字段：

```py
headers = {
    'X-Auth-Token': token
}
```

decode鸿蒙req body中的`TypedArray`类型字节字符串，利用`io.BytesIO`将其转成字节序列，再parse成图像，作为请求体传入云端部署了yolo v9模型的modelarts进行瑕疵检测

#### postgres

> 主要维护了一个image_id和图片推理结果的映射表

1. **获取所有历史记录**:

   - ```
     get_all_histories(page, limit, date_string)
     ```

     - 获取符合条件的所有历史记录
     - 支持分页（通过前端给的 `page` 和 `limit` 参数）和日期过滤（通过 `date_string` 参数）
     - 如果 `page` 为 `None`，则不进行分页，返回所有符合日期的记录

2. **插入推理结果**:

   - ```
     insert_inference_result(image_id, inference_json)
     ```

     - 向 `history` 表中插入新的推理结果
     - `image_id` 用于标识图像，`inference_json` 是推理结果的 JSON 数据

3. **获取推理结果**:

   - ```
     get_inference_json(image_id)
     ```

     - 根据 `image_id` 获取对应的推理结果
     - 返回结果为 JSON 格式的数据

#### 检测数据可视化

首先使用 OpenCV 的 `cv2.imread()` 读取指定路径的图像文件，接着遍历该图像 `detection_results` 中的检测框，逐个处理。再使用 `cv2.rectangle()` 在图像上绘制每个检测结果的边界框。最后在每个检测框的上方添加文本标签，显示瑕疵类别和置信度。

## 软件架构图

![image-20241118173414202](架构图.png)

## 技术亮点

### modelarts调用

ModelArts平台有两种鉴权模式：

> 本app采用token鉴权

- 使用AK/SK鉴权
  - AK/SK模式对敏感信息相对会更加可控一些，但是这种鉴权方式上传图片大小有限制
- 使用账号+密码获取Token，并用Token鉴权
  - 账号又分为主用户和IAM用户；可以简单把IAM用户理解为由主用户创建的，只有部分权限的子用户；token模式中我们的鉴权使用的是**IAM用户的**用户名和密码

### 真机api限制下的图片上传

由于真机api6的限制，`.net.http`原生仅支持`string` 或 `Object`作为请求体的extraData字段；且文件读写手段非常受限。网上很多基于`next `or `api9+`系列的图片读写及上传做法不适用于本app，包括但不限于：axios内置formData直接上传、httpclient二次封装简化文件上传、PhotoPicker进行图片视频的访问等等。

在精读了3.0、3.0/4.0、next系列和众多api版本的开发文档后，我发现可通过：

* 获取照片沙箱路径后，利用`fileIO.openSync`，获取读入文件标识符，同时用`fileIO.statSync` 获取文件字节大小，根据大小创先原始二进制缓冲区ArrayBuffer，再通过`uint8Array = new Uint8Array(buffer) `提供操作内存的接口，把字节数组join成string，以string的形式传给后端

* 后端拿到string，split(”,”)，map每一个index让string → int，最后用`bytes`实现`TypedArray`转成字节序列，以实现后端拿到拍摄的pcb板