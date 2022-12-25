import traceback

import requests
import wandb


class RequestHandler:
    def __init__(self, algorithm):
        self.url = "https://metarl-321f.restdb.io/rest/requests"
        self.run_id = wandb.config.exp_name
        self.algorithm = algorithm
        self.key = "14dddf034a737b5d7bb4788112c5b0c795544"

    def handle_updates(self, epoch):
        try:
            res = requests.get(self.url + "?q={\"run\":\"" + wandb.config.exp_name + "\"}", headers={"x-apikey": self.key})
            if res.status_code != 200:
                print("Could not reach database:" + str(res.status_code))
                return
            for r in res.json():
                self.handle_req(r, epoch)
            requests.delete(self.url + "/*?q={\"run\":\"" + wandb.config.exp_name + "\"}", headers={"x-apikey": self.key})
        except TimeoutError:
            print("Connection to requests database timed out.")
            return
        except ConnectionError:
            print("Error while connecting to request database.")
            return
        except Exception:
            print("Exception while connecting to request database:")
            print(traceback.format_exc())

    def handle_req(self, req, epoch):
        print(req["type"], req["data"])
        if req["type"] == "change_goal":
            env = self.algorithm.rollout_coordinator.env
            env.resample_tasks(req["data"])
        elif req["type"] == "generate_video":
            pass
        elif req["type"] == "plot_encoding":
            self.algorithm.plot_encoding_wandb(epoch)
        elif req["type"] == "plot_behavior":
            self.algorithm.plot_behavior_wandb(epoch)
        else:
            print("Unknown request:" + req["type"] + str(req["data"]))