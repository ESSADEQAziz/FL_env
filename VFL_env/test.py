def reshape_list_with_none(numbers):
    max_index = max(numbers)
    new_list = [None] * (max_index + 1)
    for i, num in enumerate(numbers):
        new_list[i + (max_index + 1 - len(numbers))] = num
    return new_list

ls = [6]
ls = reshape_list_with_none(ls)
dic = {
    '1':"a",
    '2':"b"
}
print(dic.values())
print(type([dic.values()]))