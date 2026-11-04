import general.parameters as parameters
import pdb
from general.logger import log_print
from hardware.buffer import *
from hardware.analog.array import CIM_Array
from hardware.sensor.CIP import CIP_Array
from hardware.digital import *
from general.param_runtime import P



def compute_cycle(hw_flow, Hw_config):
    
    cnt = 0
    cnt_shiftreg = 0
    delay_processing_CIM = 0
    delay_processing_CIP = 0
    first_time = 1

    delay_finished = 0
    cnt_layer = [0]
    input_img_size = list(map(int, Hw_config.get('Device level', 'Input_size').split(',')))


    idle_stage = {}
    writing_stage = {}
    reading_stage = {}
    processing_stage = {}
    finished_stage = {}
    reserved_cycle_cnt = {stage:0 for stage in hw_flow}

    indices_count = 0


    first_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, type(hw_flow[0]))]
    padding_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, padding)]
    Conv_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, (CIM_Array,CIP_Array))]
    CIM_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, CIM_Array)]
    CIP_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, CIP_Array)]
    Global_buffer_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, Global_buffer)]

    #initialize
    for stage in hw_flow:
        idle_stage[stage] = False
        if isinstance(stage, ( Global_buffer, FIFO, REG_array, InputBuf)):
            writing_stage[stage] = False
            reading_stage[stage] = False
            
        if isinstance(stage, (CIM_Array, CIP_Array)):
            processing_stage[stage] = False
            

        if isinstance(stage, (padding, ShiftAdd, BatchNormalization, Shortcut, HardTanh)):
            processing_stage[stage] = False

        finished_stage[stage] = False 

    #start compute
    log_print("Start Calculating Cycle!")
    for i in range(0,P.MAX_CYCLE_CNT):
        keys_to_delete = []

        if i%10000 == 0 :
            # print("-------------------------------")
            print("cycle:", i)

        cnt+=1

        if i == 0 :
            idle_stage[hw_flow[0]]= True
            # writing_stage[hw_flow[FIFO_indices[indices_count]]] =True

            # input_img_size = list(map(int, Hw_config.get('Device level', 'Input_size').split(',')))
            input_img_size = P.INPUT_SIZE

            cnt_padding = (input_img_size[0]/hw_flow[Conv_indices[indices_count]].network.stride[0]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)*(input_img_size[1]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)
            cnt_reg = (input_img_size[0]/hw_flow[Conv_indices[indices_count]].network.stride[0]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)*(input_img_size[1]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)
            cnt_MAC_row = input_img_size[0]
      

        if delay_processing_CIM > 0:  #delay_processing_CIM to compute cycle per CIM array
            delay_processing_CIM -= 1
            CIM_array = hw_flow[Conv_indices[indices_count]]
            CIM_array.processing_one_times(data)
            reserved_cycle_cnt[CIM_array] += 1 
            continue

        if delay_processing_CIP > 0 :
            delay_processing_CIP -= 1
            CIP_array = hw_flow[CIP_indices[0]]
            reserved_cycle_cnt[CIP_array] += 1 
            CIP_array.latency +=1
            continue

        for hw, value in idle_stage.items():
            if value:
                if isinstance(hw, (CIP_Array , CIM_Array)):
                    processing_stage[hw] = True
                    idle_stage[hw] = False
                if isinstance(hw, (padding, ShiftAdd, BatchNormalization, Shortcut, HardTanh)):
                    processing_stage[hw] = True
                    idle_stage[hw] = False
                if isinstance(hw, (FIFO, Global_buffer, REG_array, InputBuf)):
                    writing_stage[hw] = True
                    idle_stage[hw] = False

        
                   
        for stage in hw_flow:
            if isinstance(stage, (CIP_Array)):
                if  processing_stage[stage]:
                    reserved_cycle_cnt[stage] += 1     
                    stage.latency +=1

            if isinstance(stage, (CIM_Array, padding)):
                if  processing_stage[stage]:
                    reserved_cycle_cnt[stage] += 1     

            if isinstance(stage, (ShiftAdd, BatchNormalization, Shortcut, HardTanh)):
                if  processing_stage[stage]:
                    reserved_cycle_cnt[stage] += 1     
            

        for hw, value in processing_stage.items():
            if value:
                if isinstance(hw, (CIP_Array)):
                    index = hw_flow.index(hw) 
                    if first_time:
                        cycle_CIP = hw_flow[index].network.kernel_size[0] * hw_flow[index].network.kernel_size[1] * hw_flow[index].network.out_channels
                        first_time = 0
                    
                    if cycle_CIP == 0:
                        delay_processing_CIP = 0

                        index = hw_flow.index(hw) 
                        finished_stage[hw] = True
                    else:
                        delay_processing_CIP = P.CYCLE_CIP -1

                    cycle_CIP -= 1
                    hw.processing_one_times(i)
                    idle_stage[hw_flow[index + 1]] = True

                if isinstance(hw, (CIM_Array)):
                    index = hw_flow.index(hw) 
                    
                    CIM_array = hw
                    CIM_array.processing_one_times(data)
                    # log_print("Hardware",hw,"in processing stage")
                                            
                    idle_stage[hw_flow[index + 1]] = True

                    
                    processing_stage[hw] = False

                    cnt_MAC_row -= 1
                    if cnt_MAC_row == 0:
                        if P.INPUT_BUFFER == 1:
                            cnt_shiftreg = -1-((input_img_size[0]+ hw_flow[Conv_indices[indices_count]].network.padding[0]*2+1)*(hw_flow[Conv_indices[indices_count]].network.stride[0]-1)) #+1thiscycle  +2/34+2+1(1:降采样多的一列）cycle
                        cnt_MAC_row = input_img_size[0]/hw_flow[Conv_indices[indices_count]].network.stride[0]

                if isinstance(hw, (padding)):
                    
                    hw.processing_one_times(i)
                    # log_print("Hardware",hw,"in processing stage")
                    index = hw_flow.index(hw)  
                    if index < len(hw_flow) - 1:  
                        next_hw = hw_flow[index + 1]
                        idle_stage[next_hw] = True

                    cnt_padding -= 1
                    if cnt_padding == 0 :
                        processing_stage[hw] = False
                        finished_stage[hw] = True

                if isinstance(hw, (ShiftAdd, BatchNormalization, Shortcut, HardTanh)) :
                    hw.processing_one_times(i)
                    # log_print("Hardware",hw,"in processing stage")

                    processing_stage[hw] = False

                    index = hw_flow.index(hw)  
                    if index < len(hw_flow) - 1:  
                        next_hw = hw_flow[index + 1]
                        idle_stage[next_hw] = True

                
                    


        for hw, value in reading_stage.items():
            if value:
                if isinstance(hw, (FIFO)):
             
                    data = hw.dequeue()
                    # log_print("Hardware",hw,"in reading stage")
                    cnt_shiftreg +=1
                    if hw.size()== (input_img_size[0] +hw_flow[Conv_indices[indices_count]].network.padding[0]*2)*2 and not writing_stage[hw]: 
                        reading_stage[hw] = False
                        finished_stage[hw] = True

                    if cnt_shiftreg == hw_flow[Conv_indices[indices_count]].network.kernel_size[1] -1:
                        idle_stage[hw_flow[Conv_indices[indices_count]]] = True
                        if P.BIT_SERIAL:
                            delay_processing_CIM = (P.CYCLE_CIM*P.ACTIVATION_BIT*P.WEIGHT_BIT)-1
                        else: delay_processing_CIM = P.CYCLE_CIM -1
                        cnt_shiftreg -=hw_flow[Conv_indices[indices_count]].network.stride[0] 

                if isinstance(hw, (REG_array)):
                    for count in range(hw_flow[Conv_indices[indices_count]].network.kernel_size[1]):
                        data = hw.dequeue()
                    reading_stage[hw] = False

                    index = hw_flow.index(hw) 
                    next_hw = hw_flow[index + 1]
                    idle_stage[next_hw] = True
                    if P.BIT_SERIAL:
                        delay_processing_CIM = (P.CYCLE_CIM*P.ACTIVATION_BIT*P.WEIGHT_BIT)-1
                    else: delay_processing_CIM = P.CYCLE_CIM -1
                    # pdb.set_trace()
                    cnt_reg -= 1
                    if cnt_reg == 0:
                        writing_stage[hw] = False
                        finished_stage[hw] = True

                if isinstance(hw, (InputBuf)):
                    for count in range(hw_flow[Conv_indices[indices_count]].network.kernel_size[0]* hw_flow[Conv_indices[indices_count]].network.kernel_size[1]):
                        data = hw.dequeue()
                    reading_stage[hw] = False

                    index = hw_flow.index(hw) 
                    next_hw = hw_flow[index + 1]
                    idle_stage[next_hw] = True
                    if P.BIT_SERIAL:
                        delay_processing_CIM = (P.CYCLE_CIM*P.ACTIVATION_BIT*P.WEIGHT_BIT)-1
                    else: delay_processing_CIM = P.CYCLE_CIM -1
                    # pdb.set_trace()
                    cnt_reg -= 1
                    if cnt_reg == 0:
                        writing_stage[hw] = False
                        finished_stage[hw] = True
                                  
                        

        for stage in hw_flow:
            if isinstance(stage, (FIFO ,Global_buffer, REG_array, InputBuf)):
                if writing_stage[stage] or reading_stage[stage]  :
                    reserved_cycle_cnt[stage] += 1
            
        for hw, value in writing_stage.items():
            if value:  
                if isinstance(hw, (FIFO)):
                    hw.enqueue(i)
                    # log_print("Hardware",hw,"in writing stage")
                    if hw.is_full() or hw.size() >= (input_img_size[0]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)*hw_flow[Conv_indices[indices_count]].network.kernel_size[0]: #3fifo
                        reading_stage[hw]= True
                        # cnt+=2

                    if len(padding_indices) == 0 :    #if no padding layer, FIFO writing stage depend on reg_array，else depend on padding
                        cnt_padding -= 1  
                        if cnt_padding == 0:
                            writing_stage[hw] = False
                    else:
                        writing_stage[hw] = False
                        
                if isinstance(hw, (Global_buffer)):
                    hw.write(i)
                    index = hw_flow.index(hw)
                    writing_stage[hw_flow[index]] = False
                    # log_print("Hardware",hw,"in writing stage")


                if isinstance(hw, (REG_array)):
                    hw.enqueue(i)
                    if hw.size() >=  hw_flow[Conv_indices[indices_count]].network.kernel_size[0]* hw_flow[Conv_indices[indices_count]].network.kernel_size[1]:
                        reading_stage[hw] = True

                    # if len(padding_indices) == 0 :    
                    #     cnt_padding -= 1  
                    #     if cnt_padding == 0:
                    #         writing_stage[hw] = False
                    # else:
                    #     writing_stage[hw] = False

                if isinstance(hw, (InputBuf)):
                    hw.enqueue(i)
                    if hw.size() >=  hw_flow[Conv_indices[indices_count]].network.kernel_size[0]* hw_flow[Conv_indices[indices_count]].network.kernel_size[1]:
                        reading_stage[hw] = True


        if delay_finished == 1:
            finished_stage[hw_flow[Conv_indices[indices_count]]] = True 
            delay_finished = 0
        elif delay_finished == 2:
            index_delay += 1
            finished_stage[hw_flow[index_delay]] = True 
            delay_finished = 0


        for hw, value in finished_stage.items():
            if value:
                # log_print("Hardware",hw,"in finished stage")
                if isinstance(hw, (padding)):
                    keys_to_delete.append(hw)


                if isinstance(hw, (FIFO, REG_array, InputBuf)):
                    writing_stage[hw] = False
                    reading_stage[hw] = False

                    delay_finished = 1
                    keys_to_delete.append(hw)

                if isinstance(hw, (CIP_Array)):
                    processing_stage[hw] = False

                    delay_finished = 2 
                    index_delay = hw_flow.index(hw) 

                    keys_to_delete.append(hw)

                if isinstance(hw, (CIM_Array)):
                    processing_stage[hw] = False 

                    delay_finished = 2 
                    index_delay = hw_flow.index(hw)  
                    
                    keys_to_delete.append(hw)

                if isinstance(hw, (ShiftAdd, BatchNormalization, Shortcut, HardTanh)):
                    processing_stage[hw] = False

                    delay_finished = 2 
                    index_delay = hw_flow.index(hw)  

                    keys_to_delete.append(hw)



                if isinstance(hw,(Global_buffer)):
                    
                    keys_to_delete.append(hw)

                    input_img_size = [(input_img_size[0]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2 -hw_flow[Conv_indices[indices_count]].network.kernel_size[0]+1) /hw_flow[Conv_indices[indices_count]].network.stride[0], \
                                      (input_img_size[1]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2 -hw_flow[Conv_indices[indices_count]].network.kernel_size[1]+1) /hw_flow[Conv_indices[indices_count]].network.stride[0]]

                    indices_count += 1
                    if indices_count == len(Global_buffer_indices):
                        break
                
                    cnt_padding = (input_img_size[0]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)*(input_img_size[1]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)
                    cnt_reg = (input_img_size[0]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)*(input_img_size[1]+hw_flow[Conv_indices[indices_count]].network.padding[0]*2)
                    cnt_MAC_row = input_img_size[0]/hw_flow[Conv_indices[indices_count]].network.stride[0]
                
                    index = hw_flow.index(hw)  
                    if index < len(hw_flow) - 1:                 
                        next_hw = hw_flow[index + 1]
                        idle_stage[next_hw] = True
                        cnt_layer.append((cnt))
                    


        for hw in keys_to_delete:
            del finished_stage[hw]


        if not finished_stage:
            cnt_layer.append(cnt)
            log_print("\n[Summary]")
            log_print("Every layer cycle count:", cnt_layer)
            log_print("Overall system cycle count: ", cnt)
            log_print("[Cycle distribution]", reserved_cycle_cnt)
            break

    return cnt, cnt_layer

   
   