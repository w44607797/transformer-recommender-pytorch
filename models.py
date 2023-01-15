import torch
import torch.nn as nn
from din import DinAttentionLayer, din_padding_mask
from torch.nn import TransformerEncoder, LayerNorm
import torch.optim as optim
class BSTModel(nn.Module):
    def __init__(self,
                 sparse_input_length=1,
                 max_seq_length=50,
                 vocab_size_dict=None,
                 embedding_dim=512,
                 dnn_unit_list=[512, 128, 32],
                 activation='relu',
                 dropout_rate=0.2,
                 n_layers=2,
                 num_heads=8,
                 middle_units=1024,
                 training=False):
        super(BSTModel, self).__init__()

        # 1. Input layer
        # 1.1 user
        self.user_id_embedding = nn.Embedding(vocab_size_dict["user_id"] + 1, embedding_dim, padding_idx=0)
        self.gender_embedding = nn.Embedding(vocab_size_dict["gender"] + 1, embedding_dim, padding_idx=0)
        self.age_embedding = nn.Embedding(vocab_size_dict["age"] + 1, embedding_dim, padding_idx=0)
        self.item_id_embedding = nn.Embedding(vocab_size_dict["item_id"] + 1, embedding_dim, padding_idx=0)
        self.cate_id_embedding = nn.Embedding(vocab_size_dict["cate_id"] + 1, embedding_dim, padding_idx=0)
        self.num_heads = num_heads
        self.middle_units = middle_units
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        # 3. Concat layer

        self.dnn = nn.Sequential(nn.Linear(dnn_unit_list[0]*5, dnn_unit_list[1]),
                                 nn.ReLU(),nn.Dropout(dropout_rate),
                                 nn.Linear(dnn_unit_list[1], dnn_unit_list[2]),
                                 nn.ReLU(),nn.Dropout(dropout_rate))
        self.output_layer = nn.Linear(dnn_unit_list[2], 1)
        self.sigmoid = nn.Sigmoid()
        # 4. Transformer layer
        d_model = embedding_dim
        self.transformer_encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(d_model*2, self.num_heads, self.middle_units),
            num_layers=self.n_layers
        )
        self.output_din_layer = DinAttentionLayer(d_model, self.middle_units, self.dropout_rate)
        # 5. Din attention layer
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, user_id, gender, age, user_click_item_seq, user_click_cate_seq, target_item, target_cate):
        # Embedding layer
        user_id_embed = self.user_id_embedding(user_id)
        gender_embed = self.gender_embedding(gender)
        age_embed = self.age_embedding(age)
        user_click_item_seq_embed = self.item_id_embedding(user_click_item_seq)
        user_click_cate_seq_embed = self.cate_id_embedding(user_click_cate_seq)
        target_item_embed = self.item_id_embedding(target_item)
        target_cate_embed = self.cate_id_embedding(target_cate)

        # Concat layer
        other_features_concat = torch.cat([user_id_embed,
                                                 gender_embed, age_embed],
                                                dim=-1)
        # 3.1 user: sequence features
        input_transformer = torch.cat([user_click_item_seq_embed,
                                             user_click_cate_seq_embed],
                                            dim=-1)
        # 3.2 item
        input_din_query = torch.cat([target_item_embed,
                                           target_cate_embed], dim=-1)


        # Applying the encoder to a batch of input sequences
        output_tranformer = self.transformer_encoder(input_transformer)

        # 5. Din attention layer

        query = input_din_query
        keys = output_tranformer
        vecs = output_tranformer

        din_padding_mask_list = din_padding_mask(user_click_item_seq)

        output_din = self.output_din_layer.forward([query, keys, vecs, din_padding_mask_list])
        output_din = output_din.view(3, -1)

        # DNN layer
        input_dnn = torch.cat((other_features_concat, output_din), dim=-1)
        dnn_output = self.dnn(input_dnn)
        # Output layer
        output = self.output_layer(dnn_output)
        sig_output = self.sigmoid(output)
        return sig_output

    def loss(self,predict,target):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(predict, target)
        return loss


if __name__ == "__main__":
    # Initializing the input values
    user_id = torch.tensor([1, 2, 3])
    gender = torch.tensor([0, 1, 1])
    age = torch.tensor([25, 30, 35])
    user_click_item_seq = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    user_click_cate_seq = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target_item = torch.tensor([10, 11, 12])
    target_cate = torch.tensor([10, 11, 12])
    label = torch.tensor([[0.0],[0.0],[1.0]])
    vocab_size_dict = {
        "user_id": 300,
        "gender": 2,
        "age": 50,
        "item_id": 5000,
        "cate_id": 213}
    model = BSTModel(vocab_size_dict=vocab_size_dict)
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    output = model(user_id, gender, age, user_click_item_seq, user_click_cate_seq, target_item, target_cate)
    print(output)
    loss = model.loss(output,label)
    loss.backward()
    optimizer.step()
