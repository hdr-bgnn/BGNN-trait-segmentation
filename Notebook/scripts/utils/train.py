import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, rn_model, rn_loss, lambda_r=1.0, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.rn_model = rn_model
        self.rn_loss = rn_loss
        self.lambda_r = lambda_r

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        rnloss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred, rn_loss = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)
                
                rn_loss_value = rn_loss.cpu().detach().numpy()
                rnloss_meter.add(rn_loss_value)
                rnloss_logs = {'RNLoss': rnloss_meter.mean}
                logs.update(rnloss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, rn_model, rn_loss, lambda_r, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            rn_model=rn_model,
            rn_loss=rn_loss,
            lambda_r = 1.0,
        )
        self.optimizer = optimizer
        self.device = device
        self.rn_model = rn_model
        self.rn_loss = rn_loss
        self.lambda_r = lambda_r

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        # add physics loss here
        # construct the rn_inp tensor
        D = prediction[:, 0, :, :]
        Pel = prediction[:, 4, :, :]
        Pec = prediction[:, 5, :, :]
        An = prediction[:,3, :, :]
        H = prediction[:, 6, :, :]
        T = prediction[:, 2, :, :]
        
        # Dorsal-Pectoral
        rn_inp = torch.stack((H, T, D, Pec), dim=1).to(self.device)
        y_pred = self.rn_model(rn_inp)
        pred = torch.pow(torch.min(y_pred + 0.48, torch.ones(y_pred.shape).to(self.device)), 100)
        r_loss_1 = self.rn_loss(pred, torch.ones(pred.shape).to(self.device))
        # Dorsal-Pelvic
        rn_inp = torch.stack((H, T, D, Pel), dim=1).to(self.device)
        y_pred = self.rn_model(rn_inp)
        pred = torch.pow(torch.min(y_pred + 0.48, torch.ones(y_pred.shape).to(self.device)), 100)
        r_loss_2 = self.rn_loss(pred, torch.ones(pred.shape).to(self.device))
        # Dorsal-Anal
        rn_inp = torch.stack((H, T, D, An), dim=1).to(self.device)
        y_pred = self.rn_model(rn_inp)
        pred = torch.pow(torch.min(y_pred + 0.48, torch.ones(y_pred.shape).to(self.device)), 100)
        r_loss_3 = self.rn_loss(pred, torch.ones(pred.shape).to(self.device))
        
        r_loss = r_loss_1 + r_loss_2 + r_loss_3
        loss = loss + self.lambda_r * r_loss
        loss.backward()
        self.optimizer.step()
        return loss-(self.lambda_r * r_loss), prediction, r_loss


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, rn_model, rn_loss,
                 lambda_r, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            rn_model=rn_model,
            rn_loss=rn_loss,
            lambda_r = 1.0,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
            # add physics loss here
            # construct the rn_inp tensor
            D = prediction[:, 0, :, :]
            P = prediction[:, 5, :, :]
            H = prediction[:, 6, :, :]
            T = prediction[:, 2, :, :]
            rn_inp = torch.stack((H, T, D, P), dim=1).to(self.device)
            # load the rn_dorsal_to_model
            y_pred = self.rn_model(rn_inp)
            pred = torch.pow(torch.min(y_pred + 0.48, torch.ones(y_pred.shape).to(self.device)), 100)
            r_loss = self.rn_loss(pred, torch.ones(pred.shape).to(self.device))
        return loss, prediction, r_loss