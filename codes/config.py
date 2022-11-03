class cifar_config:
    batch_size = 128
    learning_rate = 0.0001
    orig_epoch = 200 
    oe_epoch = 10
    fake_epoch = 100 
    swd_epoch=10
    swd_weight = 5.0
    oe_weight = 0.5 # 1.0, 0.5

class mnist_config:
    batch_size = 128
    learning_rate = 0.0001
    orig_epoch = 50 # 300, 50
    fake_epoch = 10 # 100, 30
    swd_epoch=5
    swd_weight = 5.0
    oe_weight = 1.0 # 5.0

class attack_config:
    batch_size = 128
    learning_rate = 0.001
    att_epoch = 50 # 30
    nu = 1.0


class eval_config:
    batch_size = 128
    tau_list = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    tau_list_h = [0.5,0.6,0.7,0.8,0.9,0.95,0.99]
    save_dir="../results"
    
class cifar_names:
    orig = "cifar_orig.pth" 
    orig_oe = "cifar_orig_oe.pth"    
    orig_HE = "cifar_HE_orig.pth"    
    orig_HE_oe = "cifar_HE_orig_oe.pth"    

    ## fake net names
    fake = "cifar_fake.pth"
    fake_sm = "cifar_fake_small.pth"
    
    fake_HE = "cifar_HE_fake.pth"
    fake_HE_sm = "cifar_HE_fake_small.pth"    

    ## attack net names
    att = "cifar_att.pth"
    att_HE = "cifar_HE_att.pth"

class mnist_names:
    orig = "mnist_orig.pth" 
    orig_oe = "mnist_orig_oe.pth"    
    orig_HE = "mnist_HE_orig.pth"    
    orig_HE_oe = "mnist_HE_orig_oe.pth"    

    ## fake net names
    fake = "mnist_fake.pth"
    fake_sm = "mnist_fake_small.pth"
    
    fake_HE = "mnist_HE_fake.pth"
    fake_HE_sm = "mnist_HE_fake_small.pth"

    ## attack net names
    att = "mnist_att.pth"
    att_HE = "mnist_HE_att.pth"