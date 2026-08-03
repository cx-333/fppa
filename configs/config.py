import yaml

class DotDict(dict):
    """
    一个支持通过句点（.）访问和修改字典键值的自定义字典类。
    支持递归将嵌套的字典也转换为 DotDict。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 遍历字典，将内部的 dict 也递归转换为 DotDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                # 如果是列表，检查列表内是否有字典并转换
                self[key] = [DotDict(item) if isinstance(item, dict) else item for item in value]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"配置中不存在属性: '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"配置中不存在属性: '{key}'")


def load_yaml_config(file_path: str) -> DotDict:
    """
    读取 YAML 文件并返回支持句点访问的 DotDict 对象。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # safe_load 比 load 更安全，防止执行恶意 YAML 节点
        config_data = yaml.safe_load(f) 
    
    # 如果 YAML 为空，返回空字典
    if config_data is None:
        config_data = {}
        
    return DotDict(config_data)


# CONFIG = load_yaml_config("./configs/config.yaml")
# CONFIG = load_yaml_config("./configs/gan_config.yaml")
# print(CONFIG)

