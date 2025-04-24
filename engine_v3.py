import torch
import numpy as np
from utils.functions import evaluation,evaluation_myself
from utils.re_ranking import re_ranking, re_ranking_gpu
import os 
import shutil
import matplotlib.pyplot as plt
from PIL import Image


try:
    import wandb
except ImportError:
    wandb = None


def save_rank_n_visualizations(
    distmat,
    q_pids,
    g_pids,
    queryset,
    galleryset,
    save_dir="./rank_n_visualizations",
    rank_n=5,
    figsize=(15, 5)
):
    """
    For each query, plot the query image and top-N matched gallery images (same pid) in one image.
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    distmat = torch.tensor(distmat).float()
    num_q = distmat.size(0)

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_img_path = queryset[q_idx][0]

        # Get sorted gallery indices by distance
        ranked_indices = torch.argsort(distmat[q_idx])
        selected_gallery_paths = []

        for g_idx in ranked_indices:
            if g_pids[g_idx] == q_pid and g_idx != q_idx:
                selected_gallery_paths.append(galleryset[g_idx][0])
                if len(selected_gallery_paths) >= rank_n:
                    break

        # If no matches found, skip
        if len(selected_gallery_paths) == 0:
            continue

        # Plot query + rank-N
        fig, axes = plt.subplots(1, rank_n + 1, figsize=figsize)
        fig.suptitle(f"Query {q_idx} | PID {q_pid}", fontsize=12)

        # Plot query image
        q_img = Image.open(q_img_path).convert("RGB")
        axes[0].imshow(q_img)
        axes[0].set_title("Query")
        axes[0].axis("off")

        # Plot top-N matched gallery images
        for i, g_path in enumerate(selected_gallery_paths):
            g_img = Image.open(g_path).convert("RGB")
            axes[i + 1].imshow(g_img)
            axes[i + 1].set_title(f"Rank-{i+1}")
            axes[i + 1].axis("off")

        # Save to file
        save_path = os.path.join(save_dir, f"query_{q_idx:04d}_pid{q_pid}.jpg")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class Engine:
    def __init__(self, args, model, optimizer, scheduler, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.galleryset
        self.queryset = loader.queryset

        self.loader=loader
        self.ckpt = ckpt
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

        self.lr = 0.0
        self.device = torch.device("cpu" if args.cpu else "cuda")

        if torch.cuda.is_available():
            self.ckpt.write_log("[INFO] GPU: " + torch.cuda.get_device_name(0))

        self.ckpt.write_log(
            "[INFO] Starting from epoch {}".format(self.scheduler.last_epoch + 1)
        )

        if args.wandb and wandb is not None:
            self.wandb = True
            wandb.init(project=args.wandb_name)
        else:
            self.wandb = False

    def train(self):
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_last_lr()[0]

        if lr != self.lr:
            self.ckpt.write_log(
                "[INFO] Epoch: {}\tLearning rate: {:.2e}  ".format(epoch + 1, lr)
            )
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, d in enumerate(self.train_loader):
            inputs, labels = self._parse_data_for_train(d)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss.compute(outputs, labels)

            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log(
                "\r[INFO] [{}/{}]\t{}/{}\t{}".format(
                    epoch + 1,
                    self.args.epochs,
                    batch + 1,
                    len(self.train_loader),
                    self.loss.display_loss(batch),
                ),
                end="" if batch + 1 != len(self.train_loader) else "\n",
            )

            if self.wandb is True and wandb is not None:
                wandb.log(self.loss.get_loss_dict(batch))

        self.scheduler.step()
        self.loss.end_log(len(self.train_loader))
        # self._save_checkpoint(epoch, 0., self.ckpt.dir, is_best=True)

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckpt.write_log("\n[INFO] Test:")
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 6))

        with torch.no_grad():
            qf, query_ids, query_cams = self.extract_feature(
                self.query_loader, self.args
            )
            gf, gallery_ids, gallery_cams = self.extract_feature(
                self.test_loader, self.args
            )

        if self.args.re_rank:
            # q_g_dist = np.dot(qf, np.transpose(gf))
            # q_q_dist = np.dot(qf, np.transpose(qf))
            # g_g_dist = np.dot(gf, np.transpose(gf))
            # dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
            dist = re_ranking_gpu(qf, gf, 20, 6, 0.3)
        else:
            # cosine distance
            dist = 1 - torch.mm(qf, gf.t()).cpu().numpy()

        r, m_ap = evaluation(dist, query_ids, gallery_ids, query_cams, gallery_cams, 50)

        self.ckpt.log[-1, 0] = epoch
        self.ckpt.log[-1, 1] = m_ap
        self.ckpt.log[-1, 2] = r[0]
        self.ckpt.log[-1, 3] = r[2]
        self.ckpt.log[-1, 4] = r[4]
        self.ckpt.log[-1, 5] = r[9]
        best = self.ckpt.log.max(0)

        self.ckpt.write_log(
            "[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})".format(
                m_ap, r[0], r[2], r[4], r[9], best[0][1], self.ckpt.log[best[1][1], 0]
            ),
            refresh=True,
        )

        if not self.args.test_only:
            self._save_checkpoint(
                epoch,
                r[0],
                self.ckpt.dir,
                is_best=(self.ckpt.log[best[1][1], 0] == epoch),
            )
            self.ckpt.plot_map_rank(epoch)

        if self.wandb is True and wandb is not None:
            wandb.log(
                {
                    "mAP": m_ap,
                    "rank1": r[0],
                    "rank3": r[2],
                    "rank5": r[4],
                    "rank10": r[9],
                }
            )


    def test_myself(self):
        epoch = self.scheduler.last_epoch
        self.ckpt.write_log("\n[INFO] Test:")
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 6))
        import datetime
        now = datetime.datetime.now()
        
        with torch.no_grad():
            qf, query_ids, query_cams = self.extract_feature_myself(
                self.query_loader, self.args
            )
            gf, gallery_ids, gallery_cams = self.extract_feature_myself(
                self.test_loader, self.args
            )

        time_elapsed = (datetime.datetime.now() - now).seconds
        log = ' Time used: {} m {} s for 提取模板'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(log, end='\n')
        
        
        
        if self.args.re_rank:
            # q_g_dist = np.dot(qf, np.transpose(gf))
            # q_q_dist = np.dot(qf, np.transpose(qf))
            # g_g_dist = np.dot(gf, np.transpose(gf))
            # dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
            dist = re_ranking_gpu(qf, gf, 20, 6, 0.3)
        else:
            # cosine distance
            # from scipy.spatial.distance import cdist
            # dist = cdist(qf, gf, metric='cosine')
            dist = 1 - torch.mm(qf, gf.t()).cpu().numpy()


        save_dir = os.path.join("/data/yuesang", f'vis_epoch_1')
        print("6666666666666")
        save_rank_n_visualizations(
            dist,
            query_ids,
            gallery_ids,
            self.loader.query_for_mat,
            self.loader.gallery_for_mat,
            save_dir,
            rank_n=5
        )
        
        
        now = datetime.datetime.now()
        r, m_ap = evaluation_myself(dist, query_ids, gallery_ids, query_cams, gallery_cams, 50,self.args.eval_batch_size)
        
        time_elapsed = (datetime.datetime.now() - now).seconds
        log = ' Time used: {} m {} s for 计算指标'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(log, end='\n')
        
        
        
        
        self.ckpt.log[-1, 0] = epoch
        self.ckpt.log[-1, 1] = m_ap
        self.ckpt.log[-1, 2] = r[0]
        self.ckpt.log[-1, 3] = r[2]
        self.ckpt.log[-1, 4] = r[4]
        self.ckpt.log[-1, 5] = r[9]
        best = self.ckpt.log.max(0)

        self.ckpt.write_log(
            "[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})".format(
                m_ap, r[0], r[2], r[4], r[9], best[0][1], self.ckpt.log[best[1][1], 0]
            ),
            refresh=True,
        )

        if not self.args.test_only:
            self._save_checkpoint(
                epoch,
                r[0],
                self.ckpt.dir,
                is_best=(self.ckpt.log[best[1][1], 0] == epoch),
            )
            self.ckpt.plot_map_rank(epoch)

        if self.wandb is True and wandb is not None:
            wandb.log(
                {
                    "mAP": m_ap,
                    "rank1": r[0],
                    "rank3": r[2],
                    "rank5": r[4],
                    "rank10": r[9],
                }
            )
    
    
    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader, args):
        features = torch.FloatTensor()
        pids, camids = [], []

        
        
        for d in loader:
            inputs, pid, camid = self._parse_data_for_eval(d)
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)

            f1 = outputs.data.cpu()
            # flip
            inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)
            f2 = outputs.data.cpu()

            ff = f1 + f2
            if ff.dim() == 3:
                fnorm = torch.norm(
                    ff, p=2, dim=1, keepdim=True
                )  # * np.sqrt(ff.shape[2])
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)

            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            
            
            features = torch.cat((features, ff), 0)
            pids.extend(pid)
            camids.extend(camid)

        return features, np.asarray(pids), np.asarray(camids)

    
    def cosin(self,A, B):
        # 计算点积
        dot_product = torch.sum(A * B)
        
        # 计算每个向量的L2范数
        norm_A = torch.norm(A)
        norm_B = torch.norm(B)
        
        # 计算余弦相似度
        similarity = dot_product / (norm_A * norm_B)
        
        return similarity
    
    
    def extract_feature_myself(self, loader, args):
        features = torch.FloatTensor()
        pids, camids = [], []
        features=features.to(self.device)

        for d in loader:
            inputs, pid, camid = self._parse_data_for_eval(d)
            
            input_img = inputs.to(self.device)
            outputs = self.model(input_img).to(self.device)

            # f1 = outputs.data.cpu()
            f1 = outputs
            # flip
            inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
            input_img = inputs.to(self.device)
            outputs = self.model(input_img).to(self.device)
            # f2 = outputs.data.cpu()
            f2 = outputs

            ff = f1 + f2
            ff=ff
            if ff.dim() == 3:
                fnorm = torch.norm(
                    ff, p=2, dim=1, keepdim=True
                )  # * np.sqrt(ff.shape[2])
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)

            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            
            features = torch.cat((features, ff), 0)
            pids.extend(pid)
            camids.extend(camid)

        return features, np.asarray(pids), np.asarray(camids)


    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1

            return epoch > self.args.epochs

    # tools for reid datamanager data_v2
    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        self.ckpt.save_checkpoint(
            {
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "rank1": rank1,
                "optimizer": self.optimizer.state_dict(),
                "log": self.ckpt.log,
                # 'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best,
        )
