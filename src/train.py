import tensorflow as tf
import numpy as np
import cyclegan as model
import records
import utils
import os.path

def main():
    # check params

    # setup params

    PATH_TO_REC = "../data/"
    PATH_TO_CKPT = "./checkpoints"
    SOURCE = "rainy"
    TARGET = "sunny"
    EPOCHS = 100
    BATCH_SIZE = 1
    learning_rate = 0.0002
    beta1=0.5
    vlambda=10
    ckpt_dir = os.path.join(PATH_TO_CKPT, SOURCE+"2"+TARGET)

    #
    # Build trainer
    #

    """Build trainer. Generator G maps source image from to target image. Target discriminator aims to
    distinguish between generated target image and real target image. Similar, generator F maps
    target image to source image with respective source discriminator."""
    
    # real placeholder
    source = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    target = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])

    # generated placeholder
    source2target = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    target2source = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    
    # generators (adversarial)
    G = model.generator(source, name="G")
    F = model.generator(target, name="F")
    
    # generators (cycle consistent); read '_' as 'composed with'
    F_G = model.generator(G, name="F", reuse=True)
    G_F = model.generator(F, name="G", reuse=True)

    # discriminators
    D_Y = model.discriminator(G, name="D_Y")
    D_X = model.discriminator(F, name="D_X")
    
    D_target = model.discriminator(target, name="D_Y", reuse=True)
    D_source = model.discriminator(source, name="D_X", reuse=True)
    
    D_source2target = model.discriminator(source2target, name="D_Y", reuse=True)
    D_target2source = model.discriminator(target2source, name="D_X", reuse=True)
            
    # loss (discriminators)
    loss_D_target = tf.reduce_mean(tf.squared_difference(D_target, tf.ones_like(D_target)))
    loss_D_source2target = tf.reduce_mean(tf.square(D_source2target))
    D_Y_loss = tf.identity((loss_D_target + loss_D_source2target) / 2.0, name="D_Y_loss")
    
    loss_D_source = tf.reduce_mean(tf.squared_difference(D_source, tf.ones_like(D_source)))
    loss_D_target2source = tf.reduce_mean(tf.square(D_target2source))
    D_X_loss = tf.identity((loss_D_source + loss_D_target2source) / 2.0, name="D_X_loss")
    
    # loss (generator)
    G_loss_gan = tf.reduce_mean(tf.squared_difference(D_Y, tf.ones_like(D_Y)))
    F_loss_gan = tf.reduce_mean(tf.squared_difference(D_X, tf.ones_like(D_X)))
    cycle_loss = tf.reduce_mean(tf.abs(F_G - source)) + tf.reduce_mean(tf.abs(G_F - target))

    generator_loss = tf.identity(G_loss_gan + F_loss_gan + vlambda * cycle_loss, name="Gen_loss")

    # get training variables
    trainable_var = tf.trainable_variables()
    D_Y_var = [var for var in trainable_var if "D_Y" in var.name]
    D_X_var = [var for var in trainable_var if "D_X" in var.name]
    generator_var = [var for var in trainable_var if "F" in var.name or "G" in var.name]

    # get optimizers
    D_Y_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(D_Y_loss, var_list=D_Y_var)
    D_X_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(D_X_loss, var_list=D_X_var)
    generator_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(generator_loss,
                                                                                  var_list=generator_var)
    #
    # Load images
    #

    sess = tf.Session()

    # train images
    source_data = records.RecordProvider(sess,
                                         os.path.join(PATH_TO_REC, "train_"+SOURCE+".tfrecords"),
                                         batch_size=BATCH_SIZE)
    target_data = records.RecordProvider(sess,
                                         os.path.join(PATH_TO_REC, "train_"+TARGET+".tfrecords"),
                                         batch_size=BATCH_SIZE)

    cache_source2target = utils.ItemPool()
    cache_target2source = utils.ItemPool()

    # test images
    source_data_test = records.RecordProvider(sess,
                                              os.path.join(PATH_TO_REC, "test_"+SOURCE+".tfrecords"),
                                              batch_size=BATCH_SIZE)
    target_data_test = records.RecordProvider(sess,
                                              os.path.join(PATH_TO_REC, "test_"+TARGET+".tfrecords"),
                                              batch_size=BATCH_SIZE)

    #
    # Starting training
    #
    
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("[!] Using variables from %s" % ckpt.model_checkpoint_path)
    else:
        print("[!] Initialized variables")
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    counter = 0
    batch_epoch = min(len(source_data), len(target_data)) // BATCH_SIZE
    max_iter = EPOCHS * batch_epoch
    try:
        #while not coord.should_stop():
        for _ in range(0,max_iter+1):
            print(0)
            # prepare data
            source_batch = source_data.feed()
            target_batch = target_data.feed()
            print(1)
            generated_target, generated_source = sess.run([G, F], feed_dict={
                                                                    source: source_batch,
                                                                    target: target_batch})
            print(2)
            source2target_batch = np.array(cache_source2target(list(generated_target)))
            target2source_batch = np.array(cache_target2source(list(generated_source)))
            generated_target, generated_source = sess.run([G, F])
            print(3)
            # train generator
            _ = sess.run(generator_optim, feed_dict={source: source_batch, target: target_batch})

            # train D_Y
            _ = sess.run(D_Y_optim, feed_dict={target: target_batch, source2target: source2target_batch})

            # train D_X
            _ = sess.run(D_X_optim, feed_dict={source: source_batch, target2source: target2source_batch})
            print(4)
            # print and save
            counter += 1
            if counter % 1000 == 0:
                print("[*] Iterations passed: %s" % counter)
                save_path = saver.save(sess, os.path.join(ckpt_dir, "{:015}.ckpt".format(counter)))
                print("[*] Model saved in %s" % save_path)

            # sample test images
            if counter % 100 == 0:
                source_batch = source_data_test.feed()
                target_batch = target_data_test.feed()
                [s2t, s2t2s, t2s, t2s2t] = sess.run([G, F_G, F, G_F], feed_dict={
                                                                        source: source_batch,
                                                                        target: target_batch})
                sample = np.concatenate((source_batch, s2t, s2t2s, target, t2s, t2s2t), axis=0)

                save_dir = "../sample_while_training/"
                save_file = save_dir + SOURCE+ "2" + TARGET + "%s{:015}.jpg".format(counter)
                try:
                    utils.imwrite(utils.immerge(sample, 2, 3), save_file)
                except:
                    print("[!] Failed to save sample image to %s" % save_file)
                    # for the sake of laziness...
                    pass
            print("Passed round %s" % counter) 
            if counter > max_iter:
                print("[!] Reached %s epochs" % EPOCHS)
                coord.request_stop()

    except Exception as e:
        coord.request_stop(e)
    finally:
        print("Finished.")
        coord.request_stop()
        coord.join(threads)
        sess.close()
    
if __name__ == "__main__":
    main()

