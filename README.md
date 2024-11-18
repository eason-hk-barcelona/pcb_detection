# **腾讯客户端菁英班·大作业**

## 前端

JS FA模型开发，部署在HarmonyOS 4.0.0的真机上（api版本为6）

## 后端

后端默认使用5001端口处理`http`请求，5002端口处理`websocket`请求

运行前需按照`dotenv-example`内的信息配置好环境变量，以便后端调用`modelarts`服务

### How to Run

```
sudo docker-compose -f docker-compose.yml --compatibility up
```

### refs

[modelarts api](https://support.huaweicloud.com/intl/en-us/inference-modelarts/inference-modelarts-0018.html)