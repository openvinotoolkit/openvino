def download_pdpd_resnet50():
    import paddlehub as hub
    module = hub.Module(name="resnet_v2_50_imagenet")
    return module.directory

print(download_pdpd_resnet50())

