import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lstm_models.BertTextEncoder import BertTextEncoder
from torch.nn.utils.rnn import pack_padded_sequence

logger = logging.getLogger(__name__)

__all__ = ['GCD_CMR']

class ModalityDecomposition(nn.Module):
    def __init__(self, in_dim):
        super(ModalityDecomposition, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)

class WeightNet(nn.Module):
    def __init__(self, in_dim):
        super(WeightNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x):
        return self.layer(x)

class PrivateGlobalView(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PrivateGlobalView, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(p=0.1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)
    
class FusionGlobalView(nn.Module):
    def __init__(self, in_dim, hidden_size):
        super(FusionGlobalView, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Dropout(p=0.1),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size // 8),
            nn.Dropout(p=0.1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class FusionSequence(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FusionSequence, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layer(x)

class FeatureWeight(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureWeight, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)

class MediumClassifier(nn.Module):
    def __init__(self, in_dim):
        super(MediumClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(p=0.1),
            nn.Linear(in_dim, in_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(in_dim //2, 1)
        )

    def forward(self, x):
        return self.layer(x)

class FusionDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super(FusionDiscriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 1),
        )

    def forward(self, x):
        return self.layer(x)

class GCD_CMR(nn.Module):
    def __init__(self, args):
        super(GCD_CMR, self).__init__()
        self.aligned = args.need_data_aligned
        self.task_config = args
        self.text_model = BertTextEncoder(
            language=args.language,
            use_finetune=args.use_finetune)
        self.cosine = nn.CosineEmbeddingLoss()

        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(
            audio_in,
            args.a_lstm_hidden_size,
            args.audio_out,
            num_layers=args.a_lstm_layers,
            dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(
            video_in,
            args.v_lstm_hidden_size,
            args.video_out,
            num_layers=args.v_lstm_layers,
            dropout=args.v_lstm_dropout)

        self.concat_dim = args.text_out + args.audio_out + args.video_out
        self.hidden_size = 256
        self.modality_dims = [args.text_out, args.audio_out, args.video_out]
        self.criterion = nn.L1Loss()

        self.modality_decomposition = nn.ModuleList([ModalityDecomposition(dim) for dim in self.modality_dims])
        self.weight_net = nn.ModuleList([WeightNet(dim) for dim in self.modality_dims])
        self.common_fusion = FusionGlobalView(self.concat_dim, self.hidden_size)
        self.private_global_view = nn.ModuleList([PrivateGlobalView(self.hidden_size // 8, dim) for dim in self.modality_dims])
        self.proj_t2a = nn.Sequential(nn.Linear(self.modality_dims[0], self.modality_dims[1]))
        self.proj_t2v = nn.Sequential(nn.Linear(self.modality_dims[0], self.modality_dims[2]))
        self.proj_a2a = nn.Sequential(nn.Linear(self.modality_dims[1], self.modality_dims[1]))
        self.proj_v2v = nn.Sequential(nn.Linear(self.modality_dims[2], self.modality_dims[2]))
        self.fusion_sequence = FusionSequence(self.concat_dim, self.hidden_size)
        self.fusion_discriminator = FusionDiscriminator(self.concat_dim)
        self.feature_weight = FeatureWeight(self.concat_dim, self.hidden_size)
        self.medium_classifier = nn.ModuleList([MediumClassifier(dim) for dim in self.modality_dims])

        self.TA_MI_net = CLUBSample_group(args.text_out, args.audio_out)
        self.TV_MI_net = CLUBSample_group(args.text_out, args.video_out)
        self.VA_MI_net = CLUBSample_group(args.video_out, args.audio_out)

        self.optimizer_TA_MI_net = optim.Adam(self.TA_MI_net.parameters(), lr=args.mi_net_lr, weight_decay=args.mi_net_decay)
        self.optimizer_TV_MI_net = optim.Adam(self.TV_MI_net.parameters(), lr=args.mi_net_lr, weight_decay=args.mi_net_decay)
        self.optimizer_VA_MI_net = optim.Adam(self.VA_MI_net.parameters(), lr=args.mi_net_lr, weight_decay=args.mi_net_decay)


    def forward(self, text, audio, video, groundTruth_labels=None, training=True):
        audio, audio_lengths = audio
        video, video_lengths = video

        # 计算text_lengths
        text_lengths = torch.sum(text[:, 1, :], dim=1, keepdim=False).int()

        # 提取文本特征
        text_output = self.text_model(text)[:, 0, :]

        # 选择音频和视频模型输入
        if self.aligned:
            audio_output = self.audio_model(audio, text_lengths)
            visual_output = self.video_model(video, text_lengths)
        else:
            audio_output = self.audio_model(audio, audio_lengths)
            visual_output = self.video_model(video, video_lengths)

        if training:
            # 第一步前向传播
            self.features_MI_minimization(
                text_output, audio_output, visual_output)

        # 提取特征
        text_senti, audio_senti, visual_senti, circle_text_senti, circle_audio_senti, circle_visual_senti, text_modal, audio_modal, visual_modal, mi_loss = self.fusion_extract_features(
            text_output, audio_output, visual_output)

        # 合并特征
        recombined_features = self.recombine_features(
            circle_text_senti, circle_audio_senti, circle_visual_senti)

        modality_features = torch.cat(
            (text_modal, audio_modal, visual_modal), dim=1).unsqueeze(1)

        # 将它们在第一维度上拼接起来
        recombined_features = torch.cat(
            (recombined_features, modality_features), dim=1)

        # 加权特征表示
        attention_weights = self.feature_weight(
            recombined_features)  # (batch_size, 4, 1)
        recombined_features = recombined_features * attention_weights

        fusion_features = torch.sum(recombined_features, dim=1)

        output = self.fusion_discriminator(fusion_features)

        if training:
            # 计算modality与senti之间的损失
            modAsen_loss_text = self.diff_loss(text_modal, text_senti)
            modAsen_loss_audio = self.diff_loss(audio_modal, audio_senti)
            modAsen_loss_visual = self.diff_loss(visual_modal, visual_senti)
            modAsen_loss = (modAsen_loss_text + modAsen_loss_audio + modAsen_loss_visual) / 3

            # 计算medium_modal分类器的损失
            spc_text = self.medium_classifier[0](text_senti)
            spc_audio = self.medium_classifier[1](audio_senti)
            spc_visual = self.medium_classifier[2](visual_senti)
            spc_loss = (self.criterion(spc_text, groundTruth_labels) + self.criterion(spc_audio, groundTruth_labels) + self.criterion(
                spc_visual, groundTruth_labels)) / 3

            label_loss = self.criterion(output, groundTruth_labels)

            # 返回总损失
            return label_loss + self.task_config.diff_weight * modAsen_loss + self.task_config.spc_weight * \
                spc_loss + self.task_config.mi_weight * mi_loss, output, groundTruth_labels
        else:
            return output, groundTruth_labels


    def features_MI_minimization(self, text, audio, visual):
        for i in range(self.task_config.circle_time):
            # 分解模态
            primary_text = self.modality_decomposition[0](text)
            primary_audio = self.modality_decomposition[1](audio)
            primary_visual = self.modality_decomposition[2](visual)

            # 拼接模态
            global_view = torch.cat([text, audio, visual], dim=1)
            common_view = self.common_fusion(global_view)

            # 分离共享视图并断开梯度
            global_view_T = self.private_global_view[0](common_view).detach()
            global_view_A = self.private_global_view[1](common_view).detach()
            global_view_V = self.private_global_view[2](common_view).detach()

            # 计算互信息损失
            lld_TA_loss = -self.TA_MI_net.loglikeli(global_view_T, global_view_A)
            lld_TV_loss = -self.TV_MI_net.loglikeli(global_view_T, global_view_V)
            lld_VA_loss = -self.VA_MI_net.loglikeli(global_view_V, global_view_A)

            # 反向传播和优化互信息网络
            lld_TA_loss.backward()
            lld_TV_loss.backward()
            lld_VA_loss.backward()

            self.optimizer_TA_MI_net.step()
            self.optimizer_TV_MI_net.step()
            self.optimizer_VA_MI_net.step()

            self.optimizer_TA_MI_net.zero_grad()
            self.optimizer_TV_MI_net.zero_grad()
            self.optimizer_VA_MI_net.zero_grad()

            # 将共享视图与主要模态拼接
            input_f_text = torch.cat([global_view_T, primary_text], dim=1)
            input_f_audio = torch.cat([global_view_A, primary_audio], dim=1)
            input_f_visual = torch.cat([global_view_V, primary_visual], dim=1)

            # 计算权重
            weight_text = self.weight_net[0](input_f_text)
            weight_audio = self.weight_net[1](input_f_audio)
            weight_visual = self.weight_net[2](input_f_visual)

            # 计算循环senti特征
            circle_text_senti = primary_text * weight_text
            circle_audio_senti = primary_audio * weight_audio
            circle_visual_senti = primary_visual * weight_visual

            # 更新模态
            text = text - circle_text_senti
            audio = audio - circle_audio_senti
            visual = visual - circle_visual_senti



    def fusion_extract_features(self, text, audio, visual):
        # 初始化变量
        text_senti = audio_senti = visual_senti = 0.0
        circle_text_sentis = []
        circle_audio_sentis = []
        circle_visual_sentis = []
        mi_ta_loss = mi_tv_loss = mi_va_loss = 0.0

        # 执行融合和特征提取
        for i in range(self.task_config.circle_time):
            # 分解模态
            primary_text = self.modality_decomposition[0](text)
            primary_audio = self.modality_decomposition[1](audio)
            primary_visual = self.modality_decomposition[2](visual)

            # 拼接模态
            global_view = torch.cat([text, audio, visual], dim=1)
            common_view = self.common_fusion(global_view)

            # 分离共享视图
            global_view_T = self.private_global_view[0](common_view)
            global_view_A = self.private_global_view[1](common_view)
            global_view_V = self.private_global_view[2](common_view)

            # 将共享视图与主要模态拼接
            input_f_text = torch.cat([global_view_T, primary_text], dim=1)
            input_f_audio = torch.cat([global_view_A, primary_audio], dim=1)
            input_f_visual = torch.cat([global_view_V, primary_visual], dim=1)

            # 计算权重
            weight_text = self.weight_net[0](input_f_text)
            weight_audio = self.weight_net[1](input_f_audio)
            weight_visual = self.weight_net[2](input_f_visual)

            # 计算循环senti特征
            circle_text_senti = primary_text * weight_text
            circle_audio_senti = primary_audio * weight_audio
            circle_visual_senti = primary_visual * weight_visual

            # 存储循环senti特征
            circle_text_sentis.append(circle_text_senti)
            circle_audio_sentis.append(circle_audio_senti)
            circle_visual_sentis.append(circle_visual_senti)

            # 更新模态和senti总和
            text = text - circle_text_senti
            audio = audio - circle_audio_senti
            visual = visual - circle_visual_senti

            text_senti += circle_text_senti
            audio_senti += circle_audio_senti
            visual_senti += circle_visual_senti

            # 最小化相互信息
            mi_ta_loss += self.TA_MI_net.mi_est(global_view_T, global_view_A)
            mi_tv_loss += self.TV_MI_net.mi_est(global_view_T, global_view_V)
            mi_va_loss += self.VA_MI_net.mi_est(global_view_V, global_view_A)

        # 堆叠循环senti特征并计算总体MI损失
        circle_text_sentis = torch.stack(circle_text_sentis).transpose(0, 1)
        circle_audio_sentis = torch.stack(circle_audio_sentis).transpose(0, 1)
        circle_visual_sentis = torch.stack(circle_visual_sentis).transpose(0, 1)

        mi_loss = (mi_ta_loss + mi_tv_loss + mi_va_loss) / self.task_config.circle_time

        # 返回提取的特征和损失
        return text_senti, audio_senti, visual_senti, circle_text_sentis, circle_audio_sentis, circle_visual_sentis, text, audio, visual, mi_loss

    def recombine_features(self, circle_text_senti, circle_audio_senti, circle_visual_senti):
        # 归一化circle特征
        audio_features_normalized = F.normalize(self.proj_a2a(circle_audio_senti), dim=2)
        visual_features_normalized = F.normalize(self.proj_v2v(circle_visual_senti), dim=2)

        t2a_features = F.normalize(self.proj_t2a(circle_text_senti), dim=2)
        t2v_features = F.normalize(self.proj_t2v(circle_text_senti), dim=2)


        # 计算文本、音频和视觉特征的相似度
        text_audio_similarity = self.calculate_similarity(t2a_features, audio_features_normalized)
        text_visual_similarity = self.calculate_similarity(t2v_features, visual_features_normalized)

        # 找到每个文本特征在音频和视觉序列上余弦相似度最高的特征
        max_audio_features = self.find_max_similarity_feature(text_audio_similarity, circle_audio_senti)
        max_visual_features = self.find_max_similarity_feature(text_visual_similarity, circle_visual_senti)

        # 将文本特征与对应的音频和视觉特征拼接
        concatenated_features = torch.cat([circle_text_senti, max_audio_features, max_visual_features], dim=2)

        return concatenated_features

    def calculate_similarity(self, text_features, other_features):
        return F.softmax(torch.matmul(text_features, other_features.transpose(1, 2)), dim=2)

    def find_max_similarity_feature(self, similarity_matrix, feature_matrix):
        _, max_indices = torch.max(similarity_matrix, dim=2)  # (batch, seq)
        max_features = torch.gather(feature_matrix, 1, max_indices.unsqueeze(2).expand(-1, -1, feature_matrix.size(2)))  # (batch, seq, dim)
        return max_features 

    def diff_loss(self, input1, input2):
        # 将输入扁平化为形状 (batch_size, dimension)
        batch_size = input1.size(0)
        input1_flat = input1.view(batch_size, -1)
        input2_flat = input2.view(batch_size, -1)

        # 对输入进行零均值化
        input1_mean = torch.mean(input1_flat, dim=0, keepdim=True)
        input2_mean = torch.mean(input2_flat, dim=0, keepdim=True)
        input1_centered = input1_flat - input1_mean
        input2_centered = input2_flat - input2_mean

        # 对输入进行L2归一化
        input1_norm = torch.norm(input1_centered, p=2, dim=1, keepdim=True).detach() + 1e-6
        input1_normalized = input1_centered / input1_norm.expand_as(input1_centered)

        input2_norm = torch.norm(input2_centered, p=2, dim=1, keepdim=True).detach() + 1e-6
        input2_normalized = input2_centered / input2_norm.expand_as(input2_centered)

        # 计算归一化后的输入之间的余弦相似度损失
        similarity_matrix = input1_normalized.t().mm(input2_normalized)
        diff_loss = torch.mean(similarity_matrix.pow(2))

        return diff_loss

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: 输入维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout概率
            bidirectional: 是否使用双向LSTM
        Output:
            前向传播返回形状为 (batch_size, out_size) 的张量
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(
            in_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: 输入数据，形状为 (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(
            x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

class CLUBSample_group(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=256):
        super(CLUBSample_group, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear( hidden_size, y_dim))

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).mean()

    def mi_est(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = y_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.
