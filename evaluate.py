from re_evaluation.re_evaluation import NoOpStarts
from libs.atari_wrappers import make_atari, wrap_deepmind

if __name__=="__main__":
    env_name = 'SeaquestNoFrameskip-v4'
    env = make_atari(env_name, skip_noop=False, skip_maxskip=False, max_episode_steps=108000)
    env = wrap_deepmind(env, pytorch_img=True, episode_life=False, clip_rewards=False) 

    evalagent = NoOpStarts(
        env = env,
        env_name = env_name, 
        model_path = "models/DQN/SeaquestNoFrameskip-v4/2021-05-17_17-11-02/snapshot_80_score_6.dat",
        random_agent = False
    )
    result = evalagent.evaluate_agent()
    print("Mean: %2f, Max: %2f" %(result[0], result[1]))